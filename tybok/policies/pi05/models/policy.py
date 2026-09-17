# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 Physical
# Intelligence and The HuggingFace Inc. team. Licensed under the Apache License, Version 2.0;
# modified for TyBoK in 2026. See the NOTICE file.

"""PI05Policy: select_action / predict_action_chunk + weight loading.

Wraps ``PI05FlowMatching`` with the action-queue semantics: ``select_action`` pops
one action from an internal FIFO and only runs a full chunk inference when the
queue is empty. Weight loading matches checkpoint keys (minus a leading ``model.``)
against the module tree by parameter name; unknown keys are skipped.
"""

from __future__ import annotations

import os
from collections import deque
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from ..config import PI05Config
from .flow_matching import PI05FlowMatching, resize_with_pad_torch

ACTION = "action"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


# The checkpoint stores the VLM embedding table under the (tied) lm_head; remap it
# to ``language_model.embed_tokens`` on load.
PI05_KEY_REMAP = {
    "model.paligemma_with_expert.paligemma.lm_head.weight": (
        "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    ),
    "paligemma_with_expert.paligemma.lm_head.weight": (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    ),
}


def load_model(model: nn.Module, path: str) -> list[str]:
    """Load every ``*.safetensors`` in ``path`` into ``model`` by name.

    Checkpoint keys are usually prefixed with ``model.``; keys with no matching
    parameter are skipped and returned.
    """
    loaded, skipped = [], []
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        if "processor" in os.path.basename(file):
            continue  # pre/post processor stats are handled by preprocess.py
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_name = PI05_KEY_REMAP.get(weight_name, weight_name)
                candidates = [param_name]
                if param_name.startswith("model."):
                    candidates.append(param_name[len("model.") :])
                param = None
                for candidate in candidates:
                    try:
                        param = model.get_parameter(candidate)
                        break
                    except AttributeError:
                        continue
                if param is None:
                    skipped.append(weight_name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))
                loaded.append(weight_name)
    return skipped


class PI05Policy(nn.Module):
    def __init__(self, config: PI05Config, rtc_processor=None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self.model = PI05FlowMatching(config, rtc_processor=rtc_processor)
        self.reset()

    # ------------------------------------------------------------------ #
    # real-time chunking (RTC) support
    # ------------------------------------------------------------------ #
    def supports_rtc(self) -> bool:
        return True

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.rtc_processor.rtc_config.enabled

    def init_rtc_processor(self, rtc_processor) -> None:
        """Attach (or replace) the RTC processor on both policy and model.

        Allows a processor to be (re)created after construction.
        """
        self.rtc_processor = rtc_processor
        self.model.rtc_processor = rtc_processor

    # ------------------------------------------------------------------ #
    # action queue
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _get_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        **rtc_kwargs,
    ) -> torch.Tensor:
        # ``--pad-free`` drops the always-empty camera placeholder slots here too, via the
        # same ``live_camera_keys`` rule the captured graphs use (see PI05Config) -- the two
        # paths must build the same prefix length to be bit-identical.
        image_features = self.config.live_camera_keys(batch) if self.model.pad_free else None
        images, img_masks = self._preprocess_images(batch, image_features=image_features)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(images, img_masks, tokens, masks, noise=noise, **rtc_kwargs)

        # Unpad actions to the real action dimension
        original_action_dim = self.config.action_feature_shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        *,
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> torch.Tensor:
        """Predict a chunk of actions given environment observations.

        ``inference_delay`` / ``prev_chunk_left_over`` / ``execution_horizon`` drive
        the RTC guidance (ignored unless RTC is enabled). ``prev_chunk_left_over`` is
        the unexecuted tail of the previous chunk in model space (normalized actions),
        ``(T_prev, action_dim)`` or ``(B, T_prev, action_dim)``.
        """
        self.eval()
        return self._get_action_chunk(
            batch,
            noise,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        """Select a single action given environment observations.

        Maintains a FIFO; a full chunk is inferred only when the queue is empty.
        """
        self.eval()
        if self._rtc_enabled():
            raise ValueError(
                "RTC is not supported for select_action, use predict_action_chunk (see the reference PI05Policy)"
            )
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # ------------------------------------------------------------------ #
    # observation preparation
    # ------------------------------------------------------------------ #
    def _preprocess_images(
        self, batch: dict[str, torch.Tensor], image_features: list[str] | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Resize to 224x224 (centered padding), normalize pixels to [-1, 1].

        Batch images are [B, C, H, W] float32 in [0, 1]; missing image features are
        filled with all-(-1) images and a zero mask. ``image_features`` overrides the
        checkpoint's image slots.
        """
        images = []
        img_masks = []
        device = next(self.parameters()).device
        if image_features is None:
            image_features = self.config.image_features

        present_img_keys = [key for key in image_features if key in batch]
        missing_img_keys = [key for key in image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # handle both [B, C, H, W] and [B, H, W, C]
            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # normalize from [0, 1] to [-1, 1] as expected by SigLIP
            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # image features not present in the batch become all-(-1) images (mask 0), shaped like the
        # last real one (the guard above guarantees there is at least one)
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(images[-1]) * -1
            mask = torch.zeros_like(img_masks[-1])
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
