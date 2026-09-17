"""Pre/post processing for the SmolVLA deployment engine.

- add a batch dimension to observations
- append ``\\n`` to the task and tokenize it (``max_length=48``, pad right)
- move tensors to the target device
- (un)normalize with the dataset statistics stored next to the checkpoint

Normalization looks stats up under the *feature key* (e.g. ``action``); this
checkpoint only stores per-dataset keys (``so100.buffer.action``, ...), so state
normalization and action unnormalization are identity in practice, while the
mechanism still matches checkpoints that store stats under the feature key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

from .config import SmolVLAConfig

OBS_PREFIX = "observation."
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
ACTION = "action"

VISUAL = "VISUAL"
STATE = "STATE"
ACTION_TYPE = "ACTION"


@dataclass
class Normalizer:
    """Per-feature mean/std normalization.

    ``stats`` maps feature keys to ``{mean: Tensor, std: Tensor, ...}``.
    ``norm_map`` maps feature types to normalization modes.
    """

    norm_map: dict[str, str]
    stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    eps: float = 1e-8

    @classmethod
    def from_state_dict(
        cls, norm_map: dict[str, str], state: dict[str, torch.Tensor], eps: float = 1e-8
    ) -> "Normalizer":
        """Build from a flat safetensors state dict (``key.stat`` -> tensor)."""
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            stats.setdefault(key, {})[stat_name] = tensor.to(dtype=torch.float32)
        return cls(norm_map=norm_map, stats=stats, eps=eps)

    def _apply(self, tensor: torch.Tensor, key: str, feature_type: str, *, inverse: bool) -> torch.Tensor:
        norm_mode = self.norm_map.get(feature_type, "IDENTITY")
        if norm_mode == "IDENTITY" or key not in self.stats:
            return tensor
        if norm_mode != "MEAN_STD":
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        mean = self.stats[key].get("mean")
        std = self.stats[key].get("std")
        if mean is None or std is None:
            raise ValueError("MEAN_STD normalization mode requires mean and std stats")
        denom = std + self.eps
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / denom

    def normalize(self, tensor: torch.Tensor, key: str, feature_type: str) -> torch.Tensor:
        return self._apply(tensor, key, feature_type, inverse=False)

    def unnormalize(self, tensor: torch.Tensor, key: str, feature_type: str) -> torch.Tensor:
        return self._apply(tensor, key, feature_type, inverse=True)


def load_normalizer(checkpoint_dir: str, norm_map: dict[str, str]) -> Normalizer:
    """Load stats from ``policy_preprocessor_step_*_normalizer_processor.safetensors``."""
    state: dict[str, torch.Tensor] = {}
    pattern = os.path.join(checkpoint_dir, "policy_preprocessor_step_*_normalizer_processor.safetensors")
    import glob

    for file in sorted(glob.glob(pattern)):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return Normalizer.from_state_dict(norm_map, state)


def load_unnormalizer(checkpoint_dir: str, norm_map: dict[str, str]) -> Normalizer:
    state: dict[str, torch.Tensor] = {}
    pattern = os.path.join(checkpoint_dir, "policy_postprocessor_step_*_unnormalizer_processor.safetensors")
    import glob

    for file in sorted(glob.glob(pattern)):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return Normalizer.from_state_dict(norm_map, state)


class PreProcessor:
    """Turns a raw frame dict into the policy batch."""

    def __init__(
        self,
        config: SmolVLAConfig,
        tokenizer,
        normalizer: Normalizer,
        device: str = "cuda",
        camera_alias: dict[str, str] | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.device = device
        # Observation-key aliases applied before batching, e.g.
        # ``{"observation.images.wrist_image": "observation.images.camera2"}``.
        # The rename happens only when the target key is missing (the source is
        # consumed); empty by default, so frames must use the checkpoint's exact
        # feature keys.
        self.camera_alias = camera_alias or {}

    def _alias_images(self, frame: dict) -> dict:
        """Rename aliased camera keys in a copy of the raw frame."""
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
        "task": str} -> batch dict on `device`."""
        frame = self._alias_images(frame)
        batch: dict[str, torch.Tensor] = {}
        for key, value in frame.items():
            if not key.startswith(OBS_PREFIX):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            t = value
            # add batch dimension (1D state, 3D images)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            elif t.ndim == 3:
                t = t.unsqueeze(0)
            batch[key] = t

        task = frame.get("task", "")
        if isinstance(task, str) and not task.endswith("\n"):
            task = f"{task}\n"
        tokens, att_mask = self.tokenizer.encode_task(task)
        batch[OBS_LANGUAGE_TOKENS] = tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = att_mask

        batch = {k: v.to(self.device) for k, v in batch.items()}

        # normalize non-action observations (state stats may be absent -> identity)
        for key, value in list(batch.items()):
            if key.startswith("observation."):
                ftype = STATE if key == OBS_STATE else VISUAL
                batch[key] = self.normalizer.normalize(value, key, ftype)
        return batch


class PostProcessor:
    def __init__(self, config: SmolVLAConfig, unnormalizer: Normalizer, device: str = "cpu"):
        self.config = config
        self.unnormalizer = unnormalizer
        self.device = device

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        action = self.unnormalizer.unnormalize(action, ACTION, ACTION_TYPE)
        return action.to(self.device)
