# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""Real Time Chunking (RTC) configuration.

The attention-schedule enum is inlined.

RTC treats chunk generation as inpainting: the unexecuted tail of the previous
chunk is a prefix that guides (corrects) the next chunk's denoising trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RTCAttentionSchedule(str, Enum):
    """Prefix-attention weight schedule."""

    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference."""

    # Infrastructure
    enabled: bool = True

    # Core RTC settings
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if isinstance(self.prefix_attention_schedule, str):
            self.prefix_attention_schedule = RTCAttentionSchedule(self.prefix_attention_schedule.upper())
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RTCConfig | None":
        """Build a config from a JSON-style dict (checkpoint ``rtc_config`` block / CLI).

        Returns ``None`` when ``d`` is empty.
        """
        if not d:
            return None
        kwargs: dict[str, Any] = {}
        if "prefix_attention_schedule" in d:
            kwargs["prefix_attention_schedule"] = d["prefix_attention_schedule"]
        if "max_guidance_weight" in d:
            kwargs["max_guidance_weight"] = float(d["max_guidance_weight"])
        if "execution_horizon" in d:
            kwargs["execution_horizon"] = int(d["execution_horizon"])
        if "debug" in d:
            kwargs["debug"] = bool(d["debug"])
        if "debug_maxlen" in d:
            kwargs["debug_maxlen"] = int(d["debug_maxlen"])
        cfg = cls(**kwargs)
        if "enabled" in d:
            cfg.enabled = bool(d["enabled"])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "enabled": self.enabled,
            "prefix_attention_schedule": self.prefix_attention_schedule.value,
            "max_guidance_weight": self.max_guidance_weight,
            "execution_horizon": self.execution_horizon,
            "debug": self.debug,
            "debug_maxlen": self.debug_maxlen,
        }
