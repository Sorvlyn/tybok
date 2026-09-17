"""Configuration loading for the pi0.5 deployment engine.

Parses the checkpoint ``config.json`` and resolves the tokenizer directory from
``policy_preprocessor.json``. Plain dataclasses + ``json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# Gemma variants used by pi0.5.
GEMMA_VARIANTS: dict[str, dict[str, int]] = {
    "gemma_300m": {
        "width": 1024,
        "depth": 18,
        "mlp_dim": 4096,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
    },
    "gemma_2b": {
        "width": 2048,
        "depth": 18,
        "mlp_dim": 16_384,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 256,
    },
}


@dataclass
class GemmaSpec:
    """Decoder hyper-parameters for one of the pi0.5 Gemma stacks."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_variant(cls, variant: str) -> "GemmaSpec":
        if variant not in GEMMA_VARIANTS:
            raise ValueError(f"unknown pi05 gemma variant {variant!r}")
        return cls(**GEMMA_VARIANTS[variant])


@dataclass
class PI05Config:
    """pi0.5 policy config (mirrors the checkpoint config.json + derived fields)."""

    # Input / output structure
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    action_feature_shape: tuple[int, ...] = (7,)

    # Backbone variants
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Image preprocessing
    image_resolution: tuple[int, int] = (224, 224)  # (height, width), square
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 200

    # Flow matching
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    # Denoising sampler: only "euler" is bit-exact against the reference.
    sampler: str = "euler"

    # Normalization mapping (feature type -> mode)
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    )

    # Real-Time Chunking (RTC) config as stored in the checkpoint config.json.
    # Raw JSON dict; the engine builds an RTCConfig from it (CLI flags override).
    rtc_config: dict[str, Any] | None = None

    # Paths
    checkpoint_dir: str = ""
    # Tokenizer dir as stored in policy_preprocessor.json (absolute or relative
    # to the checkpoint dir).
    tokenizer_name: str = ""
    tokenizer_dir: str = ""

    # Derived
    _image_features: list[str] = field(default_factory=list, init=False, repr=False)
    _tokenizer_dir_override: str = field(default="", init=False, repr=False)

    # ------------------------------------------------------------------ #
    @property
    def image_features(self) -> list[str]:
        """Image feature keys in checkpoint order ("observation.images.image", ...)."""
        return self._image_features

    @property
    def real_camera_keys(self) -> list[str]:
        """Image feature keys a client should send (empty cameras excluded)."""
        return [k for k in self._image_features if "empty_camera" not in k]

    def live_camera_keys(self, batch: dict) -> list[str]:
        """``--pad-free``: the camera slots this request actually populates.

        Drops the always-empty ``empty_camera_*`` placeholder slots (which the checkpoint
        lists but no client ever sends) and any real camera missing from the frame. Both the
        eager path (``PI05Policy._get_action_chunk``) and the captured graphs
        (``PI05Engine._run_graph``) use this **single rule**, so they build the same prefix
        length -- without it the eager path keeps the 256-token placeholder block and the
        graph drops it, and the two can never be bit-identical.
        """
        return [k for k in self.real_camera_keys if k in batch]

    @property
    def paligemma(self) -> GemmaSpec:
        return GemmaSpec.from_variant(self.paligemma_variant)

    @property
    def expert(self) -> GemmaSpec:
        return GemmaSpec.from_variant(self.action_expert_variant)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, tokenizer_dir: str | None = None) -> "PI05Config":
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # VISUAL input feature keys, in checkpoint order
        image_features: list[str] = []
        for key, feat in cfg.get("input_features", {}).items():
            if feat.get("type") == "VISUAL":
                image_features.append(key)

        action_shape = tuple(cfg.get("output_features", {}).get("action", {}).get("shape", (7,)))
        resolution = cfg.get("image_resolution", [224, 224])

        tokenizer_name, tokenizer_dir_resolved = _resolve_tokenizer(checkpoint_dir, tokenizer_dir)

        cfg_obj = cls(
            chunk_size=cfg.get("chunk_size", 50),
            n_action_steps=cfg.get("n_action_steps", 50),
            max_state_dim=cfg.get("max_state_dim", 32),
            max_action_dim=cfg.get("max_action_dim", 32),
            action_feature_shape=action_shape,
            paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
            image_resolution=(int(resolution[0]), int(resolution[1])),
            empty_cameras=cfg.get("empty_cameras", 0),
            tokenizer_max_length=cfg.get("tokenizer_max_length", 200),
            num_steps=cfg.get("num_inference_steps", 10),
            min_period=cfg.get("min_period", 4e-3),
            max_period=cfg.get("max_period", 4.0),
            sampler=cfg.get("sampler", "euler"),
            normalization_mapping=cfg.get("normalization_mapping", {}),
            rtc_config=cfg.get("rtc_config"),
            checkpoint_dir=checkpoint_dir,
            tokenizer_name=tokenizer_name,
            tokenizer_dir=tokenizer_dir_resolved,
        )
        cfg_obj._image_features = image_features
        if tokenizer_dir is not None:
            cfg_obj._tokenizer_dir_override = tokenizer_dir
        return cfg_obj


def _resolve_tokenizer(checkpoint_dir: str, override: str | None) -> tuple[str, str]:
    """Resolve the tokenizer dir from policy_preprocessor.json.

    ``tokenizer_name`` may be absolute or relative to the checkpoint dir; an
    explicit ``override`` wins over both.
    """
    if override:
        return os.path.abspath(override), os.path.abspath(override)

    tokenizer_name = ""
    preprocessor_path = os.path.join(checkpoint_dir, "policy_preprocessor.json")
    if os.path.isfile(preprocessor_path):
        with open(preprocessor_path, "r", encoding="utf-8") as f:
            preprocessor = json.load(f)
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                tokenizer_name = step.get("config", {}).get("tokenizer_name", "")
                break

    if not tokenizer_name:
        raise ValueError(
            "no tokenizer_processor.tokenizer_name found in policy_preprocessor.json; pass --tokenizer-dir explicitly"
        )

    if os.path.isdir(tokenizer_name):
        return tokenizer_name, tokenizer_name
    candidate = os.path.join(checkpoint_dir, tokenizer_name)
    if os.path.isdir(candidate):
        return tokenizer_name, candidate
    raise ValueError(
        f"tokenizer dir {tokenizer_name!r} from policy_preprocessor.json does not exist "
        f"(resolved relative to {checkpoint_dir} as {candidate!r}); "
        "make the checkpoint self-contained or pass --tokenizer-dir"
    )
