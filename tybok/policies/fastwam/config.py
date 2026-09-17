"""Configuration loading for the fastWAM deployment engine.

Parses the checkpoint ``config.json`` (``FastWAMConfig`` fields + scheduler
params) and resolves three external frozen Wan2.2 component dirs: tokenizer,
text encoder (bf16, or fp8 e4m3fn+scale) and Wan VAE. Plain dataclasses +
``json``. Each dir may be absolute,
checkpoint-relative, or a CLI override (``--tokenizer-dir`` /
``--text-encoder-dir`` / ``--vae-dir``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _pick_dir(*candidates: str) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return ""


def _resolve_local(checkpoint_dir: str, value: str, override: str | None, role: str) -> str:
    """Resolve a model-id / path from config to an existing local dir.

    Order: explicit CLI override > value as absolute local dir > value as a
    directory relative to the checkpoint dir.
    """
    if override:
        resolved = os.path.abspath(os.path.expanduser(override))
        if not os.path.isdir(resolved):
            raise ValueError(f"override {role} dir {override!r} does not exist (resolved {resolved!r})")
        return resolved
    if value:
        abs_value = os.path.abspath(os.path.expanduser(value))
        if os.path.isdir(abs_value):
            return abs_value
        candidate_path = os.path.join(checkpoint_dir, value)
        if os.path.isdir(candidate_path):
            return candidate_path
    raise ValueError(
        f"{role} dir {value!r} from config.json is not available locally; "
        f"pass an explicit --{role.replace('_', '-')}-dir pointing at the local copy"
    )


def _resolve_wan_root_subdir(root: str, role: str) -> str:
    """When pointed at the Wan2.2-TI2V-5B-Diffusers *root*, pick its component dir."""
    candidates = []
    if role == "text_encoder":
        # Prefer the fp8 quantization when present (same architecture config).
        for sub in ("text_encoder_fp8", "text_encoder"):
            cand = os.path.join(root, sub)
            if os.path.isfile(os.path.join(cand, "model.safetensors")):
                return cand
        candidates.append("text_encoder")
    elif role == "vae":
        candidates.append("vae")
    for sub in candidates:
        cand = os.path.join(root, sub)
        if os.path.isdir(cand):
            return cand
    raise ValueError(f"no `{role}` sub-directory under Wan2.2 root {root!r}")


@dataclass
class FastWAMDiTSpec:
    """One DiT expert's hyper-parameters (mirrors ``video_dit_config`` / ``action_dit_config``)."""

    hidden_dim: int = 3072
    in_dim: int = 48
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 48
    num_heads: int = 24
    attn_head_dim: int = 128
    num_layers: int = 30
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    action_dim: int = 14
    action_conditioned: bool = False
    action_group_causal_mask_mode: str = "group_diagonal"
    video_attention_mask_mode: str = "first_frame_causal"
    seperated_timestep: bool = True
    fuse_vae_embedding_in_latents: bool = True
    fp32_attention: bool = True
    use_gradient_checkpointing: bool = False
    has_image_input: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any], action_dim: int) -> "FastWAMDiTSpec":
        return cls(
            hidden_dim=int(d.get("hidden_dim", 3072)),
            in_dim=int(d.get("in_dim", 48)),
            ffn_dim=int(d.get("ffn_dim", 14336)),
            freq_dim=int(d.get("freq_dim", 256)),
            text_dim=int(d.get("text_dim", 4096)),
            out_dim=int(d.get("out_dim", 48)),
            num_heads=int(d.get("num_heads", 24)),
            attn_head_dim=int(d.get("attn_head_dim", 128)),
            num_layers=int(d.get("num_layers", 30)),
            eps=float(d.get("eps", 1e-6)),
            patch_size=tuple(int(p) for p in d.get("patch_size", (1, 2, 2))),
            action_dim=int(d.get("action_dim", action_dim)),
            action_conditioned=bool(d.get("action_conditioned", False)),
            action_group_causal_mask_mode=str(d.get("action_group_causal_mask_mode", "group_diagonal")),
            video_attention_mask_mode=str(d.get("video_attention_mask_mode", "first_frame_causal")),
            seperated_timestep=bool(d.get("seperated_timestep", True)),
            fuse_vae_embedding_in_latents=bool(d.get("fuse_vae_embedding_in_latents", True)),
            fp32_attention=bool(d.get("fp32_attention", True)),
            use_gradient_checkpointing=bool(d.get("use_gradient_checkpointing", False)),
            has_image_input=bool(d.get("has_image_input", False)),
        )


@dataclass
class FastWAMSchedulerSpec:
    """Continuous flow-matching scheduler params (``infer_shift``/num steps)."""

    infer_shift: float = 5.0
    num_train_timesteps: int = 1000

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FastWAMSchedulerSpec":
        d = d or {}
        return cls(
            infer_shift=float(d.get("infer_shift", 5.0)),
            num_train_timesteps=int(d.get("num_train_timesteps", 1000)),
        )


DEFAULT_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


@dataclass
class FastWAMConfig:
    """fastWAM policy config (mirrors the checkpoint config.json + resolved dirs)."""

    action_dim: int = 14
    proprio_dim: int = 14
    action_horizon: int = 32
    n_action_steps: int = 32

    # Image geometry: concatenated camera image, stored as (height, width).
    image_size: tuple[int, int] = (384, 320)
    context_len: int = 128  # fixed text-context token count seen by the experts

    tokenizer_model_id: str = ""
    tokenizer_max_len: int = 128
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    negative_prompt: str = ""
    text_cfg_scale: float = 1.0

    num_inference_steps: int = 10
    sigma_shift: float | None = None
    inference_seed: int | None = 42
    rand_device: str = "cpu"
    tiled: bool = False

    video: FastWAMDiTSpec = field(default_factory=FastWAMDiTSpec)
    action: FastWAMDiTSpec = field(default_factory=FastWAMDiTSpec)
    action_scheduler: FastWAMSchedulerSpec = field(default_factory=FastWAMSchedulerSpec)

    # Normalization mapping (FEATURE_TYPE -> mode)
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    )

    checkpoint_dir: str = ""
    text_encoder_model_id: str = ""
    tokenizer_dir: str = ""
    text_encoder_dir: str = ""
    vae_dir: str = ""

    # Derived (filled by from_pretrained)
    _image_features: list[tuple[str, tuple[int, ...]]] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ #
    @property
    def image_feature_keys(self) -> list[str]:
        return [key for key, _ in self._image_features]

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        tokenizer_dir: str | None = None,
        text_encoder_dir: str | None = None,
        vae_dir: str | None = None,
    ) -> "FastWAMConfig":
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # keys kept in checkpoint order; fastWAM usually exposes a single
        # pre-concatenated "observation.images.image"
        image_features: list[tuple[str, tuple[int, ...]]] = []
        for key, feat in cfg.get("input_features", {}).items():
            if feat.get("type") == "VISUAL":
                image_features.append((key, tuple(int(x) for x in feat.get("shape", (3, 384, 320)))))

        action_dim = int(cfg.get("action_dim", 14))
        image_size = tuple(int(x) for x in cfg.get("image_size", (384, 320)))

        video_spec = FastWAMDiTSpec.from_dict(cfg.get("video_dit_config", {}) or {}, action_dim)
        action_spec = FastWAMDiTSpec.from_dict(cfg.get("action_dit_config", {}) or {}, action_dim)

        # text encoder id: HF repo id in released checkpoints, local dir in
        # deployment setups; resolve whichever is available locally.
        text_encoder_id = str(cfg.get("text_encoder_model_id", "") or "")
        tokenizer_id = str(cfg.get("tokenizer_model_id", "") or "")
        tokenizer_dir_resolved = _resolve_local(checkpoint_dir, tokenizer_id, tokenizer_dir, "tokenizer")
        if text_encoder_dir is None:
            te_root = _resolve_local(checkpoint_dir, text_encoder_id, None, "text_encoder")
            # Wan2.2 root dir (contains text_encoder[/fp8], vae, tokenizer, ...)
            if os.path.isdir(os.path.join(te_root, "vae")) or os.path.isdir(os.path.join(te_root, "text_encoder")):
                text_encoder_dir_resolved = _resolve_wan_root_subdir(te_root, "text_encoder")
            else:
                text_encoder_dir_resolved = te_root
        else:
            text_encoder_dir_resolved = os.path.abspath(os.path.expanduser(text_encoder_dir))
            if not os.path.isdir(text_encoder_dir_resolved):
                raise ValueError(f"text-encoder dir override {text_encoder_dir!r} does not exist")
        # VAE dir: explicit override > ``vae_model_id`` from config.json (local
        # dir, or relative to the checkpoint) > sibling `vae` of the text
        # encoder > Wan2.2 root derived from the text encoder path.
        vae_model_id = str(cfg.get("vae_model_id", "") or "")
        if vae_dir is not None:
            vae_dir_resolved = os.path.abspath(os.path.expanduser(vae_dir))
            if not os.path.isdir(vae_dir_resolved):
                raise ValueError(f"vae dir override {vae_dir!r} does not exist")
        elif vae_model_id:
            vae_dir_resolved = _resolve_local(checkpoint_dir, vae_model_id, None, "vae")
        else:
            te_parent = os.path.dirname(text_encoder_dir_resolved)
            vae_dir_resolved = _pick_dir(
                os.path.join(te_parent, "vae"),
                os.path.join(os.path.dirname(te_parent), "vae"),
                os.path.join(checkpoint_dir, "vae"),
            )
            if not vae_dir_resolved:
                raise ValueError(
                    "no Wan VAE dir found; pass --vae-dir pointing at the diffusers "
                    "AutoencoderKLWan weights (e.g. .../Wan2.2-TI2V-5B-Diffusers/vae) "
                    "or set `vae_model_id` in the checkpoint config.json"
                )

        cfg_obj = cls(
            action_dim=action_dim,
            proprio_dim=cfg.get("proprio_dim"),
            action_horizon=int(cfg.get("action_horizon", 32)),
            n_action_steps=int(cfg.get("n_action_steps", cfg.get("action_horizon", 32))),
            image_size=image_size,
            context_len=int(cfg.get("context_len", 128)),
            tokenizer_model_id=tokenizer_id,
            tokenizer_max_len=int(cfg.get("tokenizer_max_len", 128)),
            prompt_template=str(cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)),
            negative_prompt=str(cfg.get("negative_prompt", "")),
            text_cfg_scale=float(cfg.get("text_cfg_scale", 1.0)),
            num_inference_steps=int(cfg.get("num_inference_steps", 10)),
            sigma_shift=cfg.get("sigma_shift"),
            inference_seed=cfg.get("inference_seed", 42),
            rand_device=str(cfg.get("rand_device", "cpu")),
            tiled=bool(cfg.get("tiled", False)),
            video=video_spec,
            action=action_spec,
            action_scheduler=FastWAMSchedulerSpec.from_dict(cfg.get("action_scheduler")),
            normalization_mapping=cfg.get("normalization_mapping", {}),
            checkpoint_dir=checkpoint_dir,
            text_encoder_model_id=text_encoder_id,
            tokenizer_dir=tokenizer_dir_resolved,
            text_encoder_dir=text_encoder_dir_resolved,
            vae_dir=vae_dir_resolved,
        )
        cfg_obj._image_features = image_features
        if not image_features:
            raise ValueError("fastWAM checkpoint has no VISUAL input feature")
        return cfg_obj
