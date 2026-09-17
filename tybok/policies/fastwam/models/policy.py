# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2024 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""FastWAM policy wrapper: select_action / predict_action_chunk + weight load.

- ``predict_action_chunk`` runs the full ``infer_action`` pipeline once
- ``select_action`` pops one action from an internal FIFO, refilling it with a
  full ``n_action_steps`` chunk when empty

Weight loading: a plain ``safetensors`` reader matching checkpoint keys (minus the
leading ``model.``) against the module tree by parameter name.

An fp8 checkpoint (a ``*.safetensors`` containing ``*.scale_weight`` keys) takes
precedence over any bf16 files in the same dir: only it is loaded, with per-row fp8
linear weights dequantised to bf16 on the fly (``w = w8.float() * scale_weight[:, None]``).
"""

from __future__ import annotations

import glob
import logging
import os
from collections import deque

import torch
from safetensors import safe_open
from torch import nn

from ..config import FastWAMConfig

log = logging.getLogger("tybok.fastwam")

ACTION = "action"
_FP8_SUFFIX = ".scale_weight"


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def _is_fp8_file(path: str) -> bool:
    """True if the file stores fp8 weights (companion ``.scale_weight`` keys)."""
    with safe_open(path, framework="pt", device="cpu") as f:
        return any(k.endswith(_FP8_SUFFIX) for k in f.keys())


def _dequant_fp8_weight(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Row-scaled fp8 e4m3fn -> bf16 (per-row fp32 scale)."""
    if w8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn weight, got {w8.dtype}")
    if scale.ndim == 0:
        w = w8.float() * scale.float()
    elif scale.ndim == 1:
        w = w8.float() * scale.float().unsqueeze(1)
    else:
        raise ValueError(f"unsupported fp8 scale shape {tuple(scale.shape)}")
    return w.to(torch.bfloat16)


def _strip_model(name: str) -> str:
    return name.removeprefix("model.")


def fp8_checkpoint_files(path: str) -> list[str]:
    """fp8 weight files in the directory (containing ``*.scale_weight`` keys), sorted by name."""
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    files = [f for f in files if "processor" not in os.path.basename(f)]
    return [f for f in files if _is_fp8_file(f)]


def load_model(model: nn.Module, path: str) -> list[str]:
    """Load the checkpoint ``*.safetensors`` files in ``path`` into ``model`` by name.

    If any file stores fp8 weights (``*.scale_weight`` keys), only those fp8 files
    are loaded (they are self-contained); fp8 projections are loaded by target-param
    dtype:

    - target param is fp8 e4m3fn (``FP8Linear`` shell from ``fp8ify_structural``):
      the fp8 weight is copied as-is and the companion row scale goes to the
      ``*.weight_scale`` parameter -- no dequant round trip;
    - target param is bf16 (plain ``nn.Linear``): the fp8 weight is dequantised to
      bf16 on the fly (``w = w8.float() * scale_weight[:, None]``).

    Checkpoint keys are prefixed with ``model.``; keys with no matching parameter
    are skipped and reported.
    """
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    files = [f for f in files if "processor" not in os.path.basename(f)]
    fp8_files = [f for f in files if _is_fp8_file(f)]
    if fp8_files:
        files = fp8_files
        log.info(f"fp8 checkpoint detected: {[os.path.basename(f) for f in fp8_files]}")

    loaded, skipped = [], []
    scale_of: dict[str, torch.Tensor] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if any(k.endswith(_FP8_SUFFIX) for k in keys):
                for k in keys:
                    if k.endswith(_FP8_SUFFIX):
                        scale_of[_strip_model(k[: -len(_FP8_SUFFIX)])] = f.get_tensor(k)
            for weight_name in keys:
                if weight_name.endswith(_FP8_SUFFIX):
                    continue  # handled above
                param_name = _strip_model(weight_name)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    skipped.append(weight_name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if param_name.endswith(".weight") and (base := param_name[: -len(".weight")]) in scale_of:
                    if param.dtype == torch.float8_e4m3fn:
                        # fp8-resident target (FP8Linear shell): copy the fp8
                        # weight as-is and stash the row scale in weight_scale.
                        weight_loader(param, f.get_tensor(weight_name))
                        try:
                            scale_param = model.get_parameter(base + ".weight_scale")
                        except AttributeError as exc:
                            raise RuntimeError(
                                f"fp8 param {param_name} has no matching "
                                f"{base}.weight_scale; is the module a FP8Linear?"
                            ) from exc
                        scale_param.data.copy_(scale_of[base])
                    else:
                        loaded_weight = _dequant_fp8_weight(f.get_tensor(weight_name), scale_of[base])
                        weight_loader(param, loaded_weight)
                else:
                    loaded_weight = f.get_tensor(weight_name)
                    weight_loader(param, loaded_weight)
                loaded.append(weight_name)
    return skipped


class FastWAMPolicy(nn.Module):
    def __init__(self, model: nn.Module, config: FastWAMConfig):
        super().__init__()
        self.config = config
        # ``model`` is the FastWAM core module (owns the mot experts etc.)
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque = deque([], maxlen=self.config.n_action_steps)

    def _get_action_chunk(
        self,
        input_image: torch.Tensor,
        state: torch.Tensor | None,
        task: str,
        noise: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        sigma_shift: float | None = None,
    ) -> torch.Tensor:
        """Full normalized action chunk [1, action_horizon, action_dim]."""
        prompt = self.config.prompt_template.format(task=str(task))
        out = self.model.infer_action(
            input_image=input_image,
            action_horizon=self.config.action_horizon,
            prompt=prompt,
            proprio=state,
            num_inference_steps=(
                self.config.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
            ),
            sigma_shift=sigma_shift,
            seed=self.config.inference_seed,
            rand_device=self.config.rand_device,
            noise=noise,
        )
        return out["action"].unsqueeze(0)  # [1, horizon, action_dim] fp32 cpu

    @torch.no_grad()
    def predict_action_chunk(
        self,
        input_image: torch.Tensor,
        state: torch.Tensor | None,
        task: str,
        noise: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        sigma_shift: float | None = None,
    ) -> torch.Tensor:
        """Full action chunk [action_horizon, action_dim] in normalized units."""
        return self._get_action_chunk(input_image, state, task, noise, num_inference_steps, sigma_shift)[0]

    @torch.no_grad()
    def select_action(
        self,
        input_image: torch.Tensor,
        state: torch.Tensor | None,
        task: str,
        noise: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        sigma_shift: float | None = None,
    ) -> torch.Tensor:
        """Single action (action_dim,) with FIFO queue semantics."""
        if len(self._action_queue) == 0:
            actions = self._get_action_chunk(input_image, state, task, noise, num_inference_steps, sigma_shift)[
                :, : self.config.n_action_steps
            ]
            # queue per-timestep rows
            self._action_queue.extend(list(actions.transpose(0, 1)))
        return self._action_queue.popleft()
