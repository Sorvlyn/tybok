"""smolvla: model spec shared by its checks."""

from __future__ import annotations

from tests._common import ModelSpec, checkpoint_from_env

__all__ = ["SMOLVLA"]

SMOLVLA = ModelSpec(
    key="smolvla",
    checkpoint=checkpoint_from_env("smolvla"),
    # the documented best-performance tier for smolvla
    fused_flags=("tl_llm_fused_attn", "tl_vit_oproj", "tl_fused_expert"),
)
