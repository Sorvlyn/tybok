"""Model-agnostic engine interface.

Every policy backend (SmolVLA today; pi0.5 / lingbot-vla planned) implements
:class:`PolicyEngine`. The gateway / worker / protocol layers only depend on
this interface, so adding a model is a matter of dropping a new
``tybok/policies/<name>/`` package and registering it in
:mod:`tybok.registry`.

``describe()`` returns the metadata the gateway needs to process images and the
client needs to interpret actions.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

log = logging.getLogger("tybok.engine")


def compile_region(
    model: torch.nn.Module,
    owner: torch.nn.Module,
    attr: str,
    **compile_kwargs: Any,
) -> str:
    """``torch.compile`` the ``owner.attr`` entry point in place; return its **region name**.

    A compiled region is named by the attribute path it wrapped
    (``embed_prefix``, ``vlm_with_expert.forward``, ``mot.prefill_video_cache``, ``denoise_step``)
    rather than by an invented phase word, so the same vocabulary appears everywhere: the
    engine's ``--compile`` log line, :attr:`PolicyEngine.compiled_regions`, and the mode matrix
    (``tests/<model>/compile_modes.py``). Deriving the name from the wrapped attribute means a rename
    or a re-wiring cannot make the report lie.

    ``compile_kwargs`` are forwarded verbatim to ``torch.compile`` (fastwam passes
    ``dynamic=False``; the other two keep torch's default).
    """
    setattr(owner, attr, torch.compile(getattr(owner, attr), **compile_kwargs))
    prefix = next((name for name, mod in model.named_modules() if mod is owner), "")
    return f"{prefix}.{attr}" if prefix else attr


class PolicyEngine(ABC):
    """Base class for policy engines (inference only)."""

    def __init__(self, checkpoint_dir: str, device: str = "auto", **kwargs):
        self.checkpoint_dir = checkpoint_dir
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # Attribute paths of the entry points this engine wrapped with ``torch.compile`` (empty
        # when ``--compile`` is off), filled by :func:`compile_region`. Named by the same
        # vocabulary everywhere: log line, this attribute, ``tests/<model>/compile_modes.py``.
        self.compiled_regions: list[str] = []
        # Keep torch's "current device" in sync with `--device`: the fused kernels
        # query device properties (occupancy / SM count / cooperativeLaunch, see
        # kernels/coop_grid.h) from the *current* device; otherwise `--device cuda:1`
        # would size the grid using device 0's properties.
        if isinstance(device, str) and device.startswith("cuda"):
            _dev = torch.device(device)
            if _dev.index is not None:
                torch.cuda.set_device(_dev)

    # ------------------------------------------------------------------ #
    # interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Model metadata consumed by the gateway::

        {"model_type": "smolvla",
         "cameras": ["camera1", "camera2", "camera3"],   # observation keys
         "resize": [512, 512],                           # (width, height)
         "action_dim": 6,
         "chunk_size": 50}
        """

    @abstractmethod
    def select_action(self, frame: dict[str, Any], noise: torch.Tensor | None = None) -> np.ndarray:
        """Single action (action_dim,) given one observation frame."""

    @abstractmethod
    def predict_action_chunk(self, frame: dict[str, Any], noise: torch.Tensor | None = None) -> np.ndarray:
        """Full action chunk (chunk_size, action_dim) given one observation frame."""

    def supports_rtc(self) -> bool:
        """Whether this backend implements Real-Time Chunking guidance.

        RTC is pure inference math (no extra weights); backends that implement it
        override this and honour ``predict_action_chunk(..., prev_chunk_left_over=
        ..., inference_delay=..., execution_horizon=...)``.
        """
        return False

    # ------------------------------------------------------------------ #
    # shared helpers
    # ------------------------------------------------------------------ #
    def make_frame(
        self,
        images: dict[str, torch.Tensor | np.ndarray],
        state: torch.Tensor | np.ndarray | list,
        task: str,
    ) -> dict[str, Any]:
        """Assemble a raw frame dict accepted by the preprocessor.

        Images are ``(C, H, W)`` float32 tensors in ``[0, 1]``; state is a 1-D
        float vector. ``images`` keys are observation keys, e.g. ``"camera1"``
        (the ``observation.images.`` prefix is added automatically).
        """
        frame: dict[str, Any] = {}
        for key, img in images.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if not torch.is_tensor(img):
                raise TypeError(f"image {key} must be a tensor or ndarray")
            if img.ndim == 3 and img.shape[0] not in (1, 3, 4) and img.shape[2] in (1, 3, 4):
                # HWC -> CHW
                img = img.permute(2, 0, 1)
            if img.ndim == 3 and img.shape[0] not in (1, 3, 4):
                img = img.permute(2, 0, 1)
            frame[f"observation.images.{key}"] = img.float().contiguous()

        if isinstance(state, (list, tuple)):
            state = np.asarray(state, dtype=np.float32)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.astype(np.float32))
        frame["observation.state"] = state.float().contiguous()
        frame["task"] = task
        return frame

    def _warmup(self) -> None:
        """Run one dummy inference so lazy CUDA initialization / cuBLAS
        heuristics do not inflate the first real request's latency."""
        spec = self.describe()
        t0 = time.perf_counter()
        images = {
            cam: torch.zeros(3, spec["resize"][1], spec["resize"][0], dtype=torch.float32) for cam in spec["cameras"]
        }
        state = torch.zeros(8, dtype=torch.float32)
        frame = self.make_frame(images, state, "warmup")
        self.predict_action_chunk(frame)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        # With ``--compile`` this call is what compiles the inference phases (the engines install
        # the compiled entry points before the warmup), so report it as such: the first start
        # pays the full cold cost, later ones hit torch's inductor cache. fastwam logs its own.
        if getattr(self, "compile_model", False):
            log.info(
                f"--compile: phases compiled + warmup done in {time.perf_counter() - t0:.1f}s "
                f"(cold starts pay the full compile; the inductor disk cache cuts it after that)"
            )
        else:
            log.info(f"warmup done in {time.perf_counter() - t0:.1f}s")

    def _profile_frame(self, cameras: int | None = None, seed: int = 0) -> dict[str, Any]:
        """Synthetic frame for profiling/benchmarking: random images for the
        configured (or requested) cameras plus a random state."""
        spec = self.describe()
        g = torch.Generator().manual_seed(seed)
        cam_list = spec["cameras"] if cameras is None else spec["cameras"][:cameras]
        # ``spec['cameras']`` may already carry the ``observation.images.`` prefix
        # (fastwam's config keys do) -- ``make_frame`` adds it, so strip it first to stay
        # idempotent for both conventions (a no-op for bare camera names).
        images = {
            cam.split("observation.images.")[-1]: torch.rand(3, spec["resize"][1], spec["resize"][0], generator=g)
            for cam in cam_list
        }
        state = torch.rand(int(getattr(self.config, "proprio_dim", None) or 8), generator=g)
        return self.make_frame(images, state, "profile")

    def profile(self, iters: int = 5, warmup: int = 3, cameras: int | None = None, seed: int = 0) -> dict[str, Any]:
        """Wall-time latency profile of :meth:`predict_action_chunk` using the
        engine exactly as configured (graph / compile / sampler / steps).

        Synthetic frames exercise the full engine path (preprocess -> model ->
        postprocess). Model-agnostic: every backend logs the same report through its
        own ``tybok.<backend>`` logger (so it lands in the worker / serve logs) and
        returns it. Backends may override :meth:`_profile_phases` to attach a per-phase
        breakdown and :meth:`_profile_header_extra` to add fields to the header line.

        Returns a structured report::

            {"wall_ms": 93.9, "iters": 5, "cameras": 2, "spec": {...}}
        """
        frame = self._profile_frame(cameras, seed)
        for _ in range(warmup):
            self.predict_action_chunk(frame)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            self.predict_action_chunk(frame)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) / iters * 1e3
        spec = self.describe()
        cam_list = spec["cameras"] if cameras is None else spec["cameras"][:cameras]
        report: dict[str, Any] = {
            "wall_ms": wall_ms,
            "iters": iters,
            "cameras": len(cam_list),
            "spec": spec,
            "phases": self._profile_phases(iters=iters, warmup=warmup, cameras=cameras, seed=seed),
        }
        if report["phases"]:
            report["phase_total"] = sum(report["phases"].values())
        self._log_profile_report(report)
        return report

    def _profile_phases(
        self, iters: int = 5, warmup: int = 3, cameras: int | None = None, seed: int = 0
    ) -> dict[str, float]:
        """Per-phase breakdown (ms/iter) attached to the profile report.

        Model-agnostic default: none. Backends whose components can be timed
        separately override this (smolvla times its eager phases with CUDA events).
        """
        return {}

    def _profile_header_extra(self) -> str:
        """Backend-specific ``key=value`` fields appended to the profile header line."""
        return ""

    def _log_profile_report(self, report: dict[str, Any]) -> None:
        """Log the profile report through ``tybok.<model_type>`` (see :meth:`profile`)."""
        cfg = getattr(self, "config", None)
        model_type = report.get("spec", {}).get("model_type")
        prof_log = logging.getLogger(f"tybok.{model_type}") if model_type else log
        # ``num_steps`` is the smolvla/pi05 name, fastwam stores ``num_inference_steps``;
        # a missing ``sampler`` means the backend only implements euler.
        steps = getattr(cfg, "num_steps", None)
        if steps is None:
            steps = getattr(cfg, "num_inference_steps", None)
        path = (
            "cuda-graph"
            if getattr(self, "graph_enabled", False)
            else "compile"
            if getattr(self, "compile_model", False)
            else "eager"
        )
        extra = self._profile_header_extra()
        wall = report["wall_ms"]
        prof_log.info(
            f"profile: device={self.device} path={path}"
            + (f" {extra}" if extra else "")
            + f" sampler={getattr(cfg, 'sampler', 'euler')} steps={steps} "
            f"cameras={report['cameras']} iters={report['iters']}"
        )
        prof_log.info(f"profile: predict_action_chunk wall {wall:.2f} ms/iter ({1000.0 / wall:.1f} Hz)")
        phases = report.get("phases") or {}
        if phases:
            total = report.get("phase_total") or 1.0
            prof_log.info("profile: phase breakdown (ms/iter, eager components; CUDA graphs remove launch overhead):")
            for name, ms in phases.items():
                prof_log.info(f"profile:   {name:<24} {ms:>9.2f}  {ms / total * 100:>6.1f}%")
            prof_log.info(f"profile:   {'sum':<24} {report['phase_total']:>9.2f}")
