"""fastwam: the model spec and kernel tiers shared by its checks."""

from __future__ import annotations

from typing import Any

from tests._common import ModelSpec, checkpoint_from_env

__all__ = [
    "FASTWAM",
    "FUSED_COOP_TIER",
    "FUSED_SPLIT_TIER",
    "PRODUCTION_TIER",
    "TORCH_TIER",
]

FASTWAM = ModelSpec(
    key="fastwam",
    checkpoint=checkpoint_from_env("fastwam"),
)

#: Kernel tiers as ``create_engine`` / ``FastWAMEngine`` keyword flags. The fused tiers are *not*
#: bit-exact against the torch tier (they are a quantisation-drift tier), which is why the overlap
#: matrix keeps every comparison inside one tier.
TORCH_TIER: dict[str, Any] = {"video_fused": False, "action_fused": False}

FUSED_SPLIT_TIER: dict[str, Any] = {
    "video_fused": True,
    "video_fused_split": True,
    "action_fused": True,
    "action_fused_split": True,
    "action_pre_fused": True,
    "video_pre_fused": True,
}

FUSED_COOP_TIER: dict[str, Any] = {
    **FUSED_SPLIT_TIER,
    "video_fused_split": False,
    "action_fused_split": False,
}

#: The all-fused cooperative tier the deployment runs and the replay check captures.
PRODUCTION_TIER: dict[str, Any] = dict(FUSED_COOP_TIER)
