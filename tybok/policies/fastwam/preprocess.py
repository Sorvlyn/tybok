"""Pre/post processing for the fastWAM deployment engine.

- images are IDENTITY: the model maps [0, 1] -> [-1, 1] at the VAE boundary
- ``observation.state`` / action use MIN_MAX: ``2*(x - min)/(max - min) - 1``
  (per-channel dataset min/max), inverted for the predicted action chunk.

Stats come from the checkpoint's pre/post-processor ``*.safetensors`` files
(flattened keys ``<feature>.<stat>``, fp32).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

from .config import FastWAMConfig

OBS_STATE = "observation.state"
ACTION = "action"

VISUAL = "VISUAL"
STATE = "STATE"
ACTION_TYPE = "ACTION"


@dataclass
class MinMaxNormalizer:
    """MIN_MAX feature normalization matching the reference semantics.

    ``stats`` maps feature keys (``observation.state`` / ``action``) to
    ``{min: Tensor, max: Tensor}``; modes other than MIN_MAX / IDENTITY are
    rejected.
    """

    norm_map: dict[str, str]
    stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    eps: float = 1e-8

    @classmethod
    def from_state_dict(
        cls, norm_map: dict[str, str], state: dict[str, torch.Tensor], eps: float = 1e-8
    ) -> "MinMaxNormalizer":
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            stats.setdefault(key, {})[stat_name] = tensor.to(dtype=torch.float32)
        return cls(norm_map=norm_map, stats=stats, eps=eps)

    def _apply(self, tensor: torch.Tensor, key: str, feature_type: str, *, inverse: bool) -> torch.Tensor:
        norm_mode = self.norm_map.get(feature_type, "IDENTITY")
        if norm_mode == "IDENTITY" or key not in self.stats:
            return tensor
        if norm_mode != "MIN_MAX":
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        min_val = self.stats[key].get("min")
        max_val = self.stats[key].get("max")
        if min_val is None or max_val is None:
            raise ValueError("MIN_MAX normalization mode requires min and max stats")
        # the reference normalizer moves its stats to the tensor's device/dtype
        if min_val.device != tensor.device or min_val.dtype != tensor.dtype:
            self.stats[key]["min"] = min_val = min_val.to(device=tensor.device, dtype=tensor.dtype)
            self.stats[key]["max"] = max_val = max_val.to(device=tensor.device, dtype=tensor.dtype)
        denom = max_val - min_val
        # when min == max, substitute the denominator with eps (maps to -1)
        denom = torch.where(denom == 0, torch.full_like(denom, self.eps), denom)
        if inverse:
            return (tensor + 1) / 2 * denom + min_val
        return 2 * (tensor - min_val) / denom - 1

    def normalize(self, tensor: torch.Tensor, key: str, feature_type: str) -> torch.Tensor:
        return self._apply(tensor, key, feature_type, inverse=False)

    def unnormalize(self, tensor: torch.Tensor, key: str, feature_type: str) -> torch.Tensor:
        return self._apply(tensor, key, feature_type, inverse=True)


def _load_stats_files(checkpoint_dir: str, pattern: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for file in sorted(glob.glob(os.path.join(checkpoint_dir, pattern))):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def load_normalizer(checkpoint_dir: str, norm_map: dict[str, str]) -> MinMaxNormalizer:
    """Load the preprocessor stats (``policy_preprocessor_step_*_normalizer_processor.safetensors``)."""
    return MinMaxNormalizer.from_state_dict(
        norm_map, _load_stats_files(checkpoint_dir, "policy_preprocessor_step_*_normalizer_processor.safetensors")
    )


def load_unnormalizer(checkpoint_dir: str, norm_map: dict[str, str]) -> MinMaxNormalizer:
    return MinMaxNormalizer.from_state_dict(
        norm_map,
        _load_stats_files(checkpoint_dir, "policy_postprocessor_step_*_unnormalizer_processor.safetensors"),
    )


def resize_with_pad_free(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize an image tensor [B, C, H, W] (or [C, H, W]) to (target_h, target_w).

    Bilinear + align_corners=False + antialias=True. No padding: fastWAM's
    cameras are plain resized images (no pad mask in the MoT attention).
    """
    if tuple(image.shape[-2:]) == (target_h, target_w):
        return image
    lead = image.shape[:-3]
    flat = image.reshape(-1, *image.shape[-3:])
    flat = torch.nn.functional.interpolate(
        flat, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True
    )
    return flat.reshape(*lead, *flat.shape[-3:])


class PreProcessor:
    """Turns a raw engine frame into the normalized policy input tensors."""

    def __init__(
        self,
        config: FastWAMConfig,
        normalizer: MinMaxNormalizer,
        device: str = "cuda",
        camera_alias: dict[str, str] | None = None,
    ):
        self.config = config
        self.normalizer = normalizer
        self.device = device
        self.image_feature_keys = config.image_feature_keys
        self.image_size = config.image_size  # (H, W) of the concatenated image
        self.camera_alias = camera_alias or {}

    def _alias_images(self, frame: dict) -> dict:
        if not self.camera_alias:
            return frame
        out = dict(frame)
        for src, dst in self.camera_alias.items():
            if src in out and dst not in out:
                out[dst] = out.pop(src)
        return out

    def __call__(self, frame: dict) -> dict[str, torch.Tensor]:
        """frame: {"observation.images.<cam>": (C,H,W) fp32 [0,1],
                   "observation.state": (D,) fp32,
                   "task": str} -> model batch on `device`.

        Returns keys ``input_image`` ([1,3,H,W]) and ``state`` ([1,D]).
        """
        frame = self._alias_images(frame)
        target_h, target_w = self.image_size

        # Single pre-concatenated camera image is the norm; multi-camera frames
        # are resized per camera to image_size[1]//n and concatenated along width.
        images = []
        present = [k for k in self.image_feature_keys if k in frame]
        if not present:
            raise ValueError(
                f"frame is missing every image feature {self.image_feature_keys} "
                f"(have {sorted(k for k in frame if k.startswith('observation.'))})"
            )
        per_cam_w = target_w // len(self.image_feature_keys)
        for key in self.image_feature_keys:
            if key not in frame:
                raise ValueError(
                    f"frame is missing image feature {key!r}; expected keys "
                    f"{self.image_feature_keys} (use --camera-alias to rename robot cameras)"
                )
            img = frame[key]
            if not torch.is_tensor(img):
                raise TypeError(f"image {key} must be a tensor")
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim != 4 or img.shape[1] != 3:
                raise ValueError(f"image {key} must be [B,3,H,W] or [3,H,W], got {tuple(img.shape)}")
            images.append(resize_with_pad_free(img.float(), target_h, per_cam_w))
        input_image = torch.cat(images, dim=-1) if len(images) > 1 else images[0]
        input_image = input_image[:, :, :target_h, :target_w].contiguous()

        state = frame.get(OBS_STATE)
        if state is None:
            raise ValueError(f"state ({OBS_STATE}) is required for fastwam")
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = self.normalizer.normalize(state.float(), OBS_STATE, STATE)

        return {
            "input_image": input_image.to(self.device),
            "state": state.to(self.device),
            "task": frame.get("task", ""),
        }


class PostProcessor:
    def __init__(self, config: FastWAMConfig, unnormalizer: MinMaxNormalizer, device: str = "cpu"):
        self.config = config
        self.unnormalizer = unnormalizer
        self.device = device

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """Unnormalize the [-1,1] action chunk back to robot units."""
        action = self.unnormalizer.unnormalize(action.float(), ACTION, ACTION_TYPE)
        return action.to(self.device)
