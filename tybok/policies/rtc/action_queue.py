# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""Thread-safe action-chunk queue for Real-Time Chunking (RTC).

Supports RTC-enabled (replace) and RTC-disabled (append) modes, plus leftover
tracking.

In a deployment the *client* (robot control loop) owns an ``ActionQueue``: it
feeds the engine's ``prev_chunk_left_over`` from :meth:`get_left_over` and
merges each new chunk with :meth:`merge` (which skips the inference-delay
prefix). The engine itself is stateless w.r.t. RTC.
"""

from __future__ import annotations

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe action-chunk queue for real-time control.

    Keeps two sequences: ``original_queue`` (used by RTC to compute leftovers)
    and ``queue`` (processed actions ready for robot execution). RTC-enabled
    mode replaces the queue accounting for inference delay; RTC-disabled mode
    appends to maintain continuity.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize with the RTC configuration controlling queue behavior."""
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Unprocessed actions, used by RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Return the next action (cloned) or ``None`` when the queue is empty."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Return the number of unconsumed actions remaining."""
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Return True if no actions remain."""
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Return the index of the next action to be consumed."""
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Return unconsumed original actions for RTC ``prev_chunk_left_over``.

        Shape ``(remaining_steps, action_dim)``; ``None`` if no original queue
        exists. These feed the next chunk's guidance corrections.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """Return unconsumed processed actions (those being executed by the robot).

        Shape ``(remaining_steps, action_dim)``; ``None`` if no processed queue
        exists.
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ):
        """Merge a new chunk into the queue.

        Args:
            original_actions: Unprocessed policy actions ``(time_steps, action_dim)``.
            processed_actions: Post-processed actions for the robot, same shape.
            real_delay: Time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        """Replace the queue, discarding the first ``real_delay`` actions.

        Those actions were executed by the robot during inference, so they are
        stale.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].clone()
        self.queue = processed_actions[clamped_delay:].clone()

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """Append new actions, dropping already-consumed ones (non-RTC mode)."""
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_and_resolve_delays(self, real_delay: int, action_index_before_inference: int | None = None) -> int:
        """Validate delay against actions actually consumed during inference."""
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
