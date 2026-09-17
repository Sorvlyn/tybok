"""Configuration loading for the SmolVLA deployment engine.

Parses the two JSON configs that define the model:

- the checkpoint ``config.json`` (SmolVLA-level fields such as ``chunk_size``,
  ``max_state_dim``, ``num_steps``, ...)
- the VLM backbone ``config.json`` (``text_config`` + ``vision_config`` for the
  SmolVLM2 architecture)

Plain dataclasses + ``json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextConfig:
    """Gemma/Llama-style text decoder hyper-parameters."""

    hidden_size: int = 960
    intermediate_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 15
    num_key_value_heads: int = 5
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    vocab_size: int = 49280
    attention_bias: bool = False
    hidden_act: str = "silu"
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextConfig":
        return cls(
            hidden_size=d.get("hidden_size", 960),
            intermediate_size=d.get("intermediate_size", 2560),
            num_hidden_layers=d.get("num_hidden_layers", 32),
            num_attention_heads=d.get("num_attention_heads", 15),
            num_key_value_heads=d.get("num_key_value_heads", 5),
            head_dim=d.get("head_dim", 64),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 10000.0),
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            vocab_size=d.get("vocab_size", 49280),
            attention_bias=d.get("attention_bias", False),
            hidden_act=d.get("hidden_act", "silu"),
            pad_token_id=d.get("pad_token_id", 0),
        )


@dataclass
class VisionConfig:
    """SigLIP vision encoder hyper-parameters (SmolVLM variant)."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int = 512
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    num_channels: int = 3

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side**2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VisionConfig":
        return cls(
            hidden_size=d.get("hidden_size", 768),
            intermediate_size=d.get("intermediate_size", 3072),
            num_hidden_layers=d.get("num_hidden_layers", 12),
            num_attention_heads=d.get("num_attention_heads", 12),
            image_size=d.get("image_size", 512),
            patch_size=d.get("patch_size", 16),
            layer_norm_eps=d.get("layer_norm_eps", 1e-6),
            hidden_act=d.get("hidden_act", "gelu_pytorch_tanh"),
            num_channels=d.get("num_channels", 3),
        )


@dataclass
class VLMConfig:
    """SmolVLM2 backbone config (text_config + vision_config + connector)."""

    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    scale_factor: int = 4
    image_token_id: int = 49190
    pad_token_id: int = 128002

    @property
    def image_seq_len(self) -> int:
        """Number of tokens per image after the connector's pixel shuffle."""
        n = self.vision_config.num_patches  # 1024
        return n // (self.scale_factor**2)  # 64

    @classmethod
    def from_pretrained(cls, vlm_dir: str) -> "VLMConfig":
        with open(os.path.join(vlm_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", {})
        vision_cfg = cfg.get("vision_config", {})
        return cls(
            text_config=TextConfig.from_dict(text_cfg),
            vision_config=VisionConfig.from_dict(vision_cfg),
            scale_factor=cfg.get("scale_factor", 4),
            image_token_id=cfg.get("image_token_id", 49190),
            pad_token_id=cfg.get("pad_token_id", 128002),
        )


@dataclass
class SmolVLAConfig:
    """SmolVLA policy config (mirrors the fields of the checkpoint config.json)."""

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Padded state / action dims
    max_state_dim: int = 32
    max_action_dim: int = 32
    action_feature_shape: tuple[int, ...] = (6,)

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 48
    pad_language_to: str = "max_length"

    # Decoding
    num_steps: int = 10
    use_cache: bool = True
    # Denoising sampler: "euler" (reference behaviour, bit-exact) or "heun"
    # (2nd-order predictor-corrector; N steps = 2N velocity evals, not bit-exact).
    sampler: str = "euler"
    # Cache the expert cross-attention projections of the (loop-invariant)
    # prefix K/V: compute once per inference after prefill instead of every
    # denoising step. Values are bit-identical, so default-on is safe. Our own
    # optimization: lerobot only caches the *VLM-side* prefix K/V (``use_cache``)
    # and re-applies each expert layer's k_proj/v_proj on every step, i.e.
    # upstream sits permanently at this flag's "off" setting.
    cache_expert_prefix_kv: bool = True

    # Attention / expert
    attention_mode: str = "cross_attn"
    self_attn_every_n_layers: int = 2
    num_vlm_layers: int = 16
    num_expert_layers: int = -1
    expert_width_multiplier: float = 0.75
    add_image_special_tokens: bool = False
    prefix_length: int = -1

    # Time embedding (flow matching)
    min_period: float = 4e-3
    max_period: float = 4.0

    # Normalization mapping (FEATURE_TYPE -> mode)
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    )

    # Paths
    checkpoint_dir: str = ""
    vlm_model_name: str = ""
    # resolved absolute path to the VLM backbone dir (may equal ``vlm_model_name``
    # when the checkpoint config already stores an absolute path)
    vlm_dir: str = ""

    # Derived
    vlm_config: VLMConfig | None = None

    @property
    def n_expert_layers(self) -> int:
        if self.num_expert_layers is None or self.num_expert_layers <= 0:
            return self.num_vlm_layers
        return self.num_expert_layers

    @property
    def expert_hidden_size(self) -> int:
        assert self.vlm_config is not None
        return int(self.vlm_config.text_config.hidden_size * self.expert_width_multiplier)

    @property
    def image_features(self) -> list[str]:
        # order follows the checkpoint's `input_features` (camera1, camera2, ...)
        return getattr(self, "_image_features", [])

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> "SmolVLAConfig":
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        vlm_name = cfg.get("vlm_model_name", "")
        # ``vlm_model_name`` may be absolute (original checkpoints) or relative to
        # the checkpoint dir (merged checkpoints); resolve it here so the engine
        # works regardless of the current working directory.
        vlm_dir = _resolve_vlm_dir(checkpoint_dir, vlm_name)
        vlm_config = None
        if vlm_dir and os.path.isfile(os.path.join(vlm_dir, "config.json")):
            vlm_config = VLMConfig.from_pretrained(vlm_dir)

        # image feature keys in checkpoint order ("observation.images.camera1", ...)
        image_features: list[str] = []
        for key, feat in cfg.get("input_features", {}).items():
            if feat.get("type") == "VISUAL":
                image_features.append(key)

        action_shape = tuple(cfg.get("output_features", {}).get("action", {}).get("shape", (6,)))

        resize = cfg.get("resize_imgs_with_padding", [512, 512])
        return cls(
            n_obs_steps=cfg.get("n_obs_steps", 1),
            chunk_size=cfg.get("chunk_size", 50),
            n_action_steps=cfg.get("n_action_steps", 50),
            max_state_dim=cfg.get("max_state_dim", 32),
            max_action_dim=cfg.get("max_action_dim", 32),
            action_feature_shape=action_shape,
            resize_imgs_with_padding=(int(resize[0]), int(resize[1])),
            empty_cameras=cfg.get("empty_cameras", 0),
            tokenizer_max_length=cfg.get("tokenizer_max_length", 48),
            pad_language_to=cfg.get("pad_language_to", "max_length"),
            num_steps=cfg.get("num_steps", 10),
            use_cache=cfg.get("use_cache", True),
            sampler=cfg.get("sampler", "euler"),
            cache_expert_prefix_kv=cfg.get("cache_expert_prefix_kv", True),
            attention_mode=cfg.get("attention_mode", "cross_attn"),
            self_attn_every_n_layers=cfg.get("self_attn_every_n_layers", 2),
            num_vlm_layers=cfg.get("num_vlm_layers", 16),
            num_expert_layers=cfg.get("num_expert_layers", -1),
            expert_width_multiplier=cfg.get("expert_width_multiplier", 0.75),
            add_image_special_tokens=cfg.get("add_image_special_tokens", False),
            prefix_length=cfg.get("prefix_length", -1),
            min_period=cfg.get("min_period", 4e-3),
            max_period=cfg.get("max_period", 4.0),
            normalization_mapping=cfg.get("normalization_mapping", {}),
            checkpoint_dir=checkpoint_dir,
            vlm_model_name=vlm_name,
            vlm_dir=vlm_dir,
            vlm_config=vlm_config,
        )._with_image_features(image_features)

    def _with_image_features(self, image_features: list[str]) -> "SmolVLAConfig":
        self._image_features = image_features
        return self


def _resolve_vlm_dir(checkpoint_dir: str, vlm_name: str) -> str:
    """Resolve the VLM backbone dir, supporting absolute and checkpoint-relative paths."""
    if not vlm_name:
        return ""
    if os.path.isdir(vlm_name):
        return vlm_name
    candidate = os.path.join(checkpoint_dir, vlm_name)
    if os.path.isdir(candidate):
        return candidate
    return vlm_name  # keep the original value; the caller reports a clear error
