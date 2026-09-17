"""pi05: model spec shared by its checks."""

from __future__ import annotations

from tests._common import ModelSpec, checkpoint_from_env

__all__ = ["PI05"]

PI05 = ModelSpec(
    key="pi05",
    checkpoint=checkpoint_from_env("pi05"),
    # the documented best-performance tier for pi05
    fused_flags=("tl_fused_vit", "tl_llm_flash_attn", "tl_fused_expert"),
)
