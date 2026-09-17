# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""Debug information handler for Real-Time Chunking (RTC)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class DebugStep:
    """Container for debug information from a single denoising step."""

    step_idx: int = 0
    x_t: Tensor | None = None
    v_t: Tensor | None = None
    x1_t: Tensor | None = None
    correction: Tensor | None = None
    err: Tensor | None = None
    weights: Tensor | None = None
    guidance_weight: float | Tensor | None = None
    time: float | Tensor | None = None
    inference_delay: int | None = None
    execution_horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tensors: bool = False) -> dict[str, Any]:
        """Return a dict representation.

        Args:
            include_tensors: If True include raw tensors; otherwise tensor
                statistics (shape, mean, std, min, max).
        """
        result = {
            "step_idx": self.step_idx,
            "guidance_weight": (
                self.guidance_weight.item() if isinstance(self.guidance_weight, Tensor) else self.guidance_weight
            ),
            "time": self.time.item() if isinstance(self.time, Tensor) else self.time,
            "inference_delay": self.inference_delay,
            "execution_horizon": self.execution_horizon,
            "metadata": self.metadata.copy(),
        }

        tensor_fields = ["x_t", "v_t", "x1_t", "correction", "err", "weights"]
        for field_name in tensor_fields:
            tensor = getattr(self, field_name)
            if tensor is not None:
                if include_tensors:
                    result[field_name] = tensor.detach().cpu()
                else:
                    result[f"{field_name}_stats"] = {
                        "shape": tuple(tensor.shape),
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                    }

        return result


class Tracker:
    """Collects RTC debug info for recent denoising steps, keyed by time."""

    def __init__(self, enabled: bool = False, maxlen: int = 100):
        self.enabled = enabled
        self._steps = {} if enabled else None  # Keyed by time
        self._maxlen = maxlen
        self._step_counter = 0

    def reset(self) -> None:
        """Clear all recorded debug information."""
        if self.enabled and self._steps is not None:
            self._steps.clear()
        self._step_counter = 0

    @torch._dynamo.disable
    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """Record debug info for a denoising step at ``time``.

        Existing steps with the same time are updated; only non-None fields are set.
        """
        if not self.enabled:
            return

        # Convert time to a float and round to avoid float precision issues in the key
        time_value = time.item() if isinstance(time, Tensor) else time
        time_key = round(time_value, 6)

        if time_key in self._steps:
            existing_step = self._steps[time_key]
            if x_t is not None:
                existing_step.x_t = x_t.detach().clone()
            if v_t is not None:
                existing_step.v_t = v_t.detach().clone()
            if x1_t is not None:
                existing_step.x1_t = x1_t.detach().clone()
            if correction is not None:
                existing_step.correction = correction.detach().clone()
            if err is not None:
                existing_step.err = err.detach().clone()
            if weights is not None:
                existing_step.weights = weights.detach().clone()
            if guidance_weight is not None:
                existing_step.guidance_weight = guidance_weight
            if inference_delay is not None:
                existing_step.inference_delay = inference_delay
            if execution_horizon is not None:
                existing_step.execution_horizon = execution_horizon
            if metadata:
                existing_step.metadata.update(metadata)
        else:
            step = DebugStep(
                step_idx=self._step_counter,
                x_t=x_t.detach().clone() if x_t is not None else None,
                v_t=v_t.detach().clone() if v_t is not None else None,
                x1_t=x1_t.detach().clone() if x1_t is not None else None,
                correction=correction.detach().clone() if correction is not None else None,
                err=err.detach().clone() if err is not None else None,
                weights=weights.detach().clone() if weights is not None else None,
                guidance_weight=guidance_weight,
                time=time_value,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                metadata=metadata,
            )

            self._steps[time_key] = step
            self._step_counter += 1

            if self._maxlen is not None and len(self._steps) > self._maxlen:
                # Dict preserves insertion order (Python 3.7+), so the first key is oldest
                oldest_key = next(iter(self._steps))
                del self._steps[oldest_key]

    def get_all_steps(self) -> list[DebugStep]:
        """Get all recorded debug steps (may be empty if disabled)."""
        if not self.enabled or self._steps is None:
            return []

        return list(self._steps.values())

    def __len__(self) -> int:
        """Return the number of recorded debug steps."""
        if not self.enabled or self._steps is None:
            return 0
        return len(self._steps)
