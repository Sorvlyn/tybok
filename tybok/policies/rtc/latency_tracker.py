# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""Latency tracking utilities for Real-Time Chunking (RTC)."""

from __future__ import annotations

from collections import deque

import numpy as np


class LatencyTracker:
    """Tracks recent latencies and answers max/percentile queries.

    Args:
        maxlen: Sliding-window size; ``None`` keeps all samples.
    """

    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        """Add a latency sample (seconds); negatives are ignored."""
        val = float(latency)

        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float | None:
        """Return the maximum latency or None if empty."""
        return self.max_latency

    def percentile(self, q: float) -> float | None:
        """Return the q-quantile (q in [0,1]) of recorded latencies or None if empty."""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float | None:
        """Return the 95th percentile latency or None if empty."""
        return self.percentile(0.95)
