"""Pre/post processing for the pi0.5 deployment engine.

Replicates the lerobot processor pipeline:

- add a batch dimension to observations
- normalize ``observation.state`` with the dataset statistics (MEAN_STD)
- build the PaliGemma prompt ``"Task: <task>, State: <discretized state>;\\nAction: "``
  (pi0.5 feeds the state *into the text prompt*; there is no state projection)
- tokenize the prompt (``max_length=200``, right padding, leading ``<bos>``)
- move tensors to the target device
- (un)normalize the predicted action with the unnormalizer stats

Normalization stats are looked up under the feature key (``observation.state`` /
``action``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from safetensors import safe_open

from .config import PI05Config

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
    """Feature normalization (only MEAN_STD is used by the pi05 checkpoints).

    ``stats`` maps feature keys to ``{mean, std, ...}``; ``norm_map`` maps feature
    types to normalization modes.
    """

    norm_map: dict[str, str]
    stats: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    eps: float = 1e-8

    @classmethod
    def from_state_dict(
        cls, norm_map: dict[str, str], state: dict[str, torch.Tensor], eps: float = 1e-8
    ) -> "Normalizer":
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
        # move stats to the tensor's device/dtype
        if mean.device != tensor.device or mean.dtype != tensor.dtype:
            self.stats[key]["mean"] = mean = mean.to(device=tensor.device, dtype=tensor.dtype)
            self.stats[key]["std"] = std = std.to(device=tensor.device, dtype=tensor.dtype)
        denom = std + self.eps
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / denom

    def normalize(self, tensor: torch.Tensor, key: str, feature_type: str) -> torch.Tensor:
        return self._apply(tensor, key, feature_type, inverse=False)

    def unnormalize(self, tensor: torch.Tensor, key: str, feature_type: str) -> torch.Tensor:
        return self._apply(tensor, key, feature_type, inverse=True)


def _load_stats_files(checkpoint_dir: str, pattern: str) -> dict[str, torch.Tensor]:
    import glob

    state: dict[str, torch.Tensor] = {}
    for file in sorted(glob.glob(os.path.join(checkpoint_dir, pattern))):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def load_normalizer(checkpoint_dir: str, norm_map: dict[str, str]) -> Normalizer:
    """Load the normalizer stats (``policy_preprocessor_step_*_normalizer_processor.safetensors``)."""
    return Normalizer.from_state_dict(
        norm_map, _load_stats_files(checkpoint_dir, "policy_preprocessor_step_*_normalizer_processor.safetensors")
    )


def load_unnormalizer(checkpoint_dir: str, norm_map: dict[str, str]) -> Normalizer:
    return Normalizer.from_state_dict(
        norm_map, _load_stats_files(checkpoint_dir, "policy_postprocessor_step_*_unnormalizer_processor.safetensors")
    )


def build_prompt(task: str, normalized_state: torch.Tensor) -> str:
    """Build the PaliGemma prompt.

    The (MEAN_STD-normalized) state is discretized into 256 bins in [-1, 1] and
    stringified into the prompt: ``"Task: <task>, State: <int ...>;\\nAction: "``.
    """
    state_np = normalized_state.detach().cpu().numpy()
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discretized = np.digitize(state_np, bins=bins) - 1
    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized.tolist()))
    return f"Task: {cleaned_text}, State: {state_str};\nAction: "


class PreProcessor:
    """Turns a raw frame dict into the policy batch."""

    def __init__(
        self,
        config: PI05Config,
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
        # ``{"observation.images.wrist_image": "observation.images.image2"}``.
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
            if t.ndim in (1, 3):
                t = t.unsqueeze(0)
            batch[key] = t

        # normalize state first: the prompt is built from the normalized state
        if OBS_STATE in batch:
            batch[OBS_STATE] = self.normalizer.normalize(batch[OBS_STATE], OBS_STATE, STATE)

        state = batch.get(OBS_STATE)
        if state is None:
            raise ValueError("state is required for PI05")
        task = frame.get("task", "")
        if not isinstance(task, str):
            raise ValueError(f"task must be a string, got {type(task).__name__}")
        prompt = build_prompt(task, state[0] if state.ndim == 2 else state)
        tokens, att_mask = self.tokenizer.encode_prompt(prompt)
        batch[OBS_LANGUAGE_TOKENS] = tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = att_mask

        return {k: v.to(self.device) for k, v in batch.items()}


class PostProcessor:
    def __init__(self, config: PI05Config, unnormalizer: Normalizer, device: str = "cpu"):
        self.config = config
        self.unnormalizer = unnormalizer
        self.device = device

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        action = self.unnormalizer.unnormalize(action, ACTION, ACTION_TYPE)
        return action.to(self.device)
