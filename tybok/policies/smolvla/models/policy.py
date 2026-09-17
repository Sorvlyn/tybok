# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""SmolVLAPolicy: select_action / predict_action_chunk + weight loading.

``select_action`` pops one action from an internal FIFO and only triggers a full
``chunk_size``-step inference when the queue is empty.

Weights are loaded from ``safetensors`` with checkpoint keys (minus the leading
``model.``) matched against the module tree by parameter name; unknown keys
(e.g. the VLM ``lm_head``) are skipped.
"""

from __future__ import annotations

import os
from collections import deque
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from ..config import SmolVLAConfig
from .flow_matching import VLAFlowMatching, pad_vector, resize_with_pad

ACTION = "action"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
OBS_STATE = "observation.state"


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str) -> list[str]:
    """Load every ``*.safetensors`` in ``path`` into ``model`` by name.

    Checkpoint keys are expected to carry a leading ``model.`` prefix; keys with
    no matching parameter (e.g. ``vlm.lm_head``) are skipped and returned.
    """
    loaded, skipped = [], []
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        if "processor" in os.path.basename(file):
            continue  # pre/post processor stats are handled by preprocess.py
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # checkpoints store the state dict with a leading ``model.``
                # prefix; try the full path first, then without it.
                candidates = [weight_name]
                if weight_name.startswith("model."):
                    candidates.append(weight_name[len("model.") :])
                param = None
                for param_name in candidates:
                    try:
                        param = model.get_parameter(param_name)
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


class SmolVLAPolicy(nn.Module):
    def __init__(self, config: SmolVLAConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.model = VLAFlowMatching(config, tokenizer=tokenizer)
        self.reset()

    # ------------------------------------------------------------------ #
    # action queue
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _get_action_chunk(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # Unpad actions
        original_action_dim = self.config.action_feature_shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        """Select a single action given environment observations.

        Maintains a FIFO; a full chunk is inferred only when the queue is empty.
        """
        self.eval()
        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    # ------------------------------------------------------------------ #
    # observation preparation
    # ------------------------------------------------------------------ #
    def prepare_images(self, batch):
        """Resize to 512x512 (+ left/top padding), normalize pixels to [-1, 1]."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                # target is stored as (width, height); helper expects (height, width)
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )
            # Normalize from [0, 1] to [-1, 1] as expected by SigLIP
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch as fully-(-1) images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch):
        """Pad state to max_state_dim."""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state
