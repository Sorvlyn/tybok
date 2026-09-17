"""fastWAM DiT-core CUDA graph (production; enabled by ``FastWAMEngine``'s ``--graph``).

Captures ``FastWAM._denoise_core`` and the Wan VAE frame encode as one fixed-shape GPU replay
with no host sync inside the capture; ``FastWAM.infer_action`` routes the core here when set.
"""

from __future__ import annotations

import logging
import time

import torch

from ....graph import CaptureSession, GraphRunner

log = logging.getLogger("tybok.fastwam")

# The three model components the captured graph bakes in, as ``<role>=<module path>``: VAE
# frame encoder, video (prefill) expert, action (denoising) expert -- all owned by the MoT.
_GRAPH_COMPONENTS = "vision encoder=vae, llm backbone=mot.mixtures.video, action expert=mot.mixtures.action"


class FastWAMGraphRunner:
    """One CUDA graph per (steps, sigma_shift, input shapes) for fastwam's ``_denoise_core``.

    The whole core -- prefill x step-0 fork/join included -- is captured as one multi-stream
    graph, so a replay is a single ``graph.replay()``, bit-exact with the eager branch it
    recorded. Lazy capture and the side-stream input copy come from :mod:`tybok.graph`.
    """

    def __init__(self, model, device="cuda", warmup_iters: int = 3, overlap: bool = False):
        self.model = model
        self.device = torch.device(device)
        self.warmup_iters = int(warmup_iters)
        self.overlap = bool(overlap)
        self._capture_grid: tuple[int, int, int] | None = None
        self._sess = CaptureSession(device, warmup_iters=self.warmup_iters)
        # Key: ``(1, steps, sigma_shift, dtypes/shapes)``; the leading ``1`` is the model-level
        # image count, always 1 because fastwam's VAE sees one pre-concatenated frame.
        self._runner = GraphRunner(self._capture, device, label="fastwam DiT core", logger=log)
        # ``overlap`` selects the ``_denoise_core`` branch that ``_capture`` records, so the
        # replayed graph stays bit-exact with the matching eager path.
        model.overlap = self.overlap

    # ------------------------------------------------------------------ #
    @property
    def entries(self) -> dict:
        return self._runner.entries

    def keys(self):
        return self._runner.keys()

    def __len__(self) -> int:
        return len(self._runner)

    def __contains__(self, key) -> bool:
        return key in self._runner

    @property
    def _graphs(self) -> dict:
        """Back-compat alias for the diagnostics that read ``runner._graphs``."""
        return self._runner.entries

    @property
    def num_graphs(self) -> int:
        return len(self._runner)

    @property
    def num_captures(self) -> int:
        return len(self._runner)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(latents, img, ctx, steps, sigma_shift) -> tuple:
        return (
            1,  # model-level image count (one concatenated frame), see __init__
            int(steps),
            None if sigma_shift is None else float(sigma_shift),
            str(latents.dtype),
            tuple(latents.shape),
            # graph input: the captured VAE encode builds ``first_frame_latents`` from it
            tuple(img.shape),
            str(img.dtype),
            tuple(ctx.shape),
            str(ctx.dtype),
        )

    # ------------------------------------------------------------------ #
    def run(
        self,
        latents_action: torch.Tensor,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
    ) -> torch.Tensor:
        """Copy the inputs, replay, return the final ``latents_action`` (static buffer).

        ``input_image`` is the already-normalized ``[1,3,H,W]`` frame (multiples of 16); the
        captured graph runs the VAE encode on it, so callers must NOT pre-encode.
        """
        latents_action = latents_action.detach().to(self.device)
        input_image = input_image.detach().to(self.device)
        context = context.detach().to(self.device)
        context_mask = context_mask.detach().to(self.device)
        key = self._key(latents_action, input_image, context, num_inference_steps, sigma_shift)
        entry = self._runner.entry(key, (latents_action, input_image, context, context_mask))
        bufs = entry["bufs"]
        with self._runner.copy_inputs():
            bufs["lat"].copy_(latents_action)
            bufs["img"].copy_(input_image)
            bufs["ctx"].copy_(context)
            bufs["mask"].copy_(context_mask)
        self._runner.wait_inputs()
        entry["graph"].replay()
        return bufs["out"]

    # ------------------------------------------------------------------ #
    def _video_blocks(self):
        return list(self.model.mot.mixtures["video"].blocks)

    def _set_capture_grid(self, grid: tuple[int, int, int]) -> None:
        """Preset the video token grid so capture has no ``grid_sizes.tolist()`` sync."""
        self._capture_grid = tuple(int(v) for v in grid)
        attn_runner = getattr(self.model, "_video_fused_attn_runner", None)
        if attn_runner is not None:
            attn_runner.set_capture_grid(self._capture_grid)
            attn_runner._capturing = True
        else:
            for blk in self._video_blocks():
                blk._capture_grid = self._capture_grid

    def _clear_capture_grid(self) -> None:
        """Undo :meth:`_set_capture_grid`; must run on every exit path after a capture."""
        attn_runner = getattr(self.model, "_video_fused_attn_runner", None)
        if attn_runner is not None:
            attn_runner._capturing = False
        else:
            for blk in self._video_blocks():
                blk._capture_grid = None

    def _ensure_overlap_resources(self) -> None:
        """Create the overlap branch's side stream / events before capture.

        The overlap branch's fork/join ops are capturable, but the stream and event
        **handles** must already exist when ``torch.cuda.graph`` starts, since resource
        creation is not a capture-safe op (warmup usually builds them; this covers the rest).
        """
        model = self.model
        num_layers = int(model.mot.num_layers)
        if model._overlap_stream is None:
            model._overlap_stream = torch.cuda.Stream(device=self.device)
        if model._overlap_events is None or len(model._overlap_events) != num_layers:
            model._overlap_events = [torch.cuda.Event() for _ in range(num_layers)]

    def _grid_for(self, first_frame_latents: torch.Tensor) -> tuple[int, int, int]:
        patch_t, patch_h, patch_w = (int(v) for v in self.model.video_expert.patch_size)
        return (
            int(first_frame_latents.shape[2]) // patch_t,
            int(first_frame_latents.shape[3]) // patch_h,
            int(first_frame_latents.shape[4]) // patch_w,
        )

    # ------------------------------------------------------------------ #
    def _capture(self, key, request):
        """Build the static buffers from the first request of this shape, then capture.

        ``request`` is ``(latents, image, ctx, mask)``; the ``empty_like`` buffers must hold
        real data before the warm-up, since an uninitialized ``bool`` mask is invalid.
        """
        latents, image, ctx, mask = request
        steps, sigma_shift = int(key[1]), key[2]
        index = len(self._runner) + 1
        t0 = time.perf_counter()
        bufs = {
            "lat": torch.empty_like(latents),
            "img": torch.empty_like(image),
            "ctx": torch.empty_like(ctx),
            "mask": torch.empty_like(mask),
            "out": torch.empty_like(latents),
        }
        # the warmup / capture must run on real data (mask is bool: empty_like is uninitialized)
        bufs["lat"].copy_(latents)
        bufs["img"].copy_(image)
        bufs["ctx"].copy_(ctx)
        bufs["mask"].copy_(mask)

        schedule = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=self.device, dtype=latents.dtype, shift_override=sigma_shift
        )

        def core(validate: bool) -> None:
            # the VAE encode runs inside the capture, on the static image buffer (no host data)
            frame_latents = self.model._encode_input_image_latents(bufs["img"])
            latents_action = self.model._denoise_core(
                bufs["lat"],
                frame_latents,
                bufs["ctx"],
                bufs["mask"],
                num_inference_steps=steps,
                sigma_shift=sigma_shift,
                schedule=schedule,
                validate=validate,
            )
            bufs["out"].copy_(latents_action)

        # one eager encode to size the capture grid, which the graph itself cannot report
        with torch.no_grad():
            probe_latents = self.model._encode_input_image_latents(bufs["img"])
        self._set_capture_grid(self._grid_for(probe_latents))
        del probe_latents
        if self.overlap:
            self._ensure_overlap_resources()
        attn_runner = getattr(self.model, "_video_fused_attn_runner", None)
        try:
            # eager warmup: populates the device-side caches, JITs kernels and settles the
            # cuBLAS/cuDNN heuristics, so nothing lazy can happen inside the capture.
            with torch.inference_mode():
                for _ in range(self.warmup_iters):
                    core(False)
                warmup_stream = torch.cuda.Stream(device=self.device)
                warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(warmup_stream):
                    for _ in range(2):
                        core(False)
                torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
                torch.cuda.synchronize()

                if attn_runner is not None:
                    attn_runner._capturing = True
                try:
                    # validate=False: no .item() D2H sync during capture
                    graph = self._sess.capture(lambda: core(False))
                finally:
                    if attn_runner is not None:
                        attn_runner._capturing = False
        finally:
            self._clear_capture_grid()
        torch.cuda.synchronize()

        entry = {
            "graph": graph,
            "bufs": bufs,
            # RULE: every tensor the captured graph reads must be held by this runner
            # (``bufs`` / ``keepalive``), never only by a local closure -> memory reuse.
            "keepalive": (schedule[0], schedule[1]),
        }
        log.info(
            f"fastwam CUDA graph #{index} captured in {time.perf_counter() - t0:.2f}s "
            f"(steps={steps}, overlap={self.overlap}, latents={tuple(latents.shape)}): "
            f"{_GRAPH_COMPONENTS}"
        )
        return entry
