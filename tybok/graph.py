"""Shared CUDA-graph plumbing for the three policy backends.

Why this module exists
----------------------
All three backends now capture the **same unit**: one graph (or one graph per ``(shape, steps)``
key) of the model's own inference core, with the model's ``prefill_layer`` x ``step0_layer``
fusion -- where the backend has one -- living *inside* the model as plain two-stream code. That
is what makes the captured graph a single ``graph.replay()`` at request time: the fork/join is
baked into the graph by CUDA stream capture instead of being replayed layer by layer.

So what lives in the backends is only the model-specific part -- which entry point to capture
and which static buffers it needs (``fastwam``: ``_denoise_core``; ``smolvla``:
``sample_actions``; ``pi05``: ``sample_actions`` with the language-length key). What the three
*do* share, and what lives here, is the surrounding lifecycle:

1. **the request -> graph-key convention**, with the number of model-level image inputs
   first, so ``--graph-cameras 2,3`` means the same thing everywhere
   (:func:`camera_counts`). smolvla/pi05 key on the per-camera ViT slots (2 or 3 images);
   fastwam's VAE sees one pre-concatenated frame, so its image count is always ``1`` and
   ``--graph-cameras`` is a documented no-op there rather than a silent one.
2. **the capture scaffolding**: warm up on a side stream (settle cuBLAS/cuDNN heuristics
   and workspace allocation, and let the model create its side stream / events outside the
   capture), then capture under ``inference_mode`` (:class:`CaptureSession`).
3. **the key -> entry registry**: lazy capture on first sight, startup pre-capture that
   tolerates a per-key failure, and the "every key failed -> disable the graph path"
   decision the engines make from :meth:`GraphRunner.precapture` (:class:`GraphRunner`).
4. **the fork/join corpus copy** shared by all three: input copies are issued on a
   dedicated stream and joined with one event before the replay
   (:meth:`GraphRunner.copy_inputs`).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import torch

__all__ = [
    "CaptureSession",
    "GRAPH_CAMERAS_HELP",
    "GraphRunner",
    "camera_counts",
]

log = logging.getLogger("tybok.graph")

GRAPH_CAMERAS_HELP = (
    "comma-separated image counts to pre-capture at startup, e.g. '2,3' "
    "(default: the configured image count; other counts are captured lazily)"
)


# --------------------------------------------------------------------------- #
# key convention
# --------------------------------------------------------------------------- #
def camera_counts(
    graph_cameras: Iterable[int] | None,
    *,
    default: int,
    max_images: int,
) -> tuple[int, ...]:
    """Normalise ``--graph-cameras`` into the image counts to pre-capture at startup.

    A gateway sending 2 or 3 cameras needs both shapes ready, hence
    ``--graph-cameras 2,3``; without the flag the configured count is pre-captured and any
    other count is captured lazily on first sight. Counts outside ``1..max_images`` are
    dropped (with the caller logging the clamp), and the result is deduplicated and sorted
    so the capture order is deterministic.

    ``max_images`` is the number of model-level image inputs: the per-camera ViT slots for
    smolvla/pi05, and ``1`` for fastwam (single pre-concatenated frame).
    """
    lo, hi = 1, max(1, int(max_images))
    counts = sorted({int(n) for n in (graph_cameras or ()) if lo <= int(n) <= hi})
    if counts:
        return tuple(counts)
    return (min(max(1, int(default)), hi),)


# --------------------------------------------------------------------------- #
# capture scaffolding
# --------------------------------------------------------------------------- #
class CaptureSession:
    """Warm-up + capture scaffolding, shared by the three backends.

    Every backend must warm up *before* capturing: cuBLAS/cuDNN pick their kernels and
    allocate their workspaces on the first few calls, and the capture bakes whichever
    kernel/workspace the warm-up left behind. The warm-up runs on a **side stream** that is
    joined back, so it cannot interleave with the capture stream, and both the warm-up and
    the capture run under ``inference_mode`` -- the static buffers were allocated under it,
    and the captured in-place writes must not mix autograd state.
    """

    def __init__(self, device: str | torch.device, warmup_iters: int = 3):
        self.device = torch.device(device)
        self.warmup_iters = int(warmup_iters)

    def prime(self, fn: Callable[[], Any], iters: int | None = None) -> None:
        """Run ``fn`` a few times on a side stream and join the stream back."""
        iters = self.warmup_iters if iters is None else int(iters)
        main = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(main)
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(iters):
                fn()
        main.wait_stream(stream)

    @staticmethod
    def capture(fn: Callable[[], Any]) -> torch.cuda.CUDAGraph:
        """Capture ``fn`` into a graph (under ``inference_mode``)."""
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            fn()
        return graph

    def prime_and_capture(self, fn: Callable[[], Any]) -> torch.cuda.CUDAGraph:
        self.prime(fn)
        return self.capture(fn)

    def sync(self) -> None:
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
class GraphRunner:
    """``key -> entry`` registry plus the per-request copy/replay primitives.

    ``capture_fn(key, request)`` builds and captures one entry (whatever shape the backend
    wants: a single graph, a pair, or a per-layer graph dict) and returns it; this class owns
    the rest: lazy capture on first sight, startup pre-capture, and the side-stream input
    copy. ``request`` is the per-request input the backend may need to build the static
    buffers (fastwam fills them with the real tensors before the warm-up; smolvla/pi05 only
    need the key's shape and pass ``None``).

    ``key`` is a tuple whose **first element is the number of model-level image inputs**,
    so ``--graph-cameras`` maps onto it uniformly (see :func:`camera_counts`); the remaining
    elements are backend-specific (pi05 appends its language bucket, fastwam its denoise
    steps / sigma shift / tensor shapes).
    """

    def __init__(
        self,
        capture_fn: Callable[[Hashable, Any], Any],
        device: str | torch.device,
        *,
        label: str = "graph",
        logger: logging.Logger | None = None,
    ):
        self.device = torch.device(device)
        self.label = label
        self._capture_fn = capture_fn
        self._log = logger or log
        self._entries: dict[Hashable, Any] = {}
        # copy/replay sync primitives, created on first use (resource creation is not
        # capture-safe, so they must not exist *inside* a capture).
        self.copy_stream: torch.cuda.Stream | None = None
        self.ev_in: torch.cuda.Event | None = None

    # -- registry ---------------------------------------------------------- #
    def __bool__(self) -> bool:
        return bool(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self):
        return self._entries.keys()

    @property
    def entries(self) -> dict[Hashable, Any]:
        """The raw ``key -> entry`` map (diagnostics read it directly)."""
        return self._entries

    def values(self):
        return self._entries.values()

    def get(self, key: Hashable, default: Any = None) -> Any:
        return self._entries.get(key, default)

    def set(self, key: Hashable, entry: Any) -> None:
        self._entries[key] = entry

    def clear(self) -> None:
        self._entries.clear()

    def entry(self, key: Hashable, request: Any = None) -> Any:
        """The entry for ``key``, captured lazily on first sight."""
        captured = self._entries.get(key)
        if captured is None:
            captured = self._capture_fn(key, request)
            self._entries[key] = captured
        return captured

    def precapture(self, keys: Iterable[Hashable]) -> bool:
        """Capture ``keys`` now (startup); True if at least one succeeded.

        A key that fails is logged and skipped so the other shapes keep working; the caller
        disables the graph path (``graph_enabled = False``) when the return value is False.
        """
        captured_any = False
        for key in keys:
            try:
                self.set(key, self._capture_fn(key, None))
                captured_any = True
            except Exception as e:  # noqa: BLE001 - a failed shape is a result, not a crash
                self._log.warning(
                    f"CUDA {self.label} capture for {_fmt_key(key)} failed ({e}); that shape will fall back to eager"
                )
        return captured_any

    # -- per-request input copy -------------------------------------------- #
    @contextlib.contextmanager
    def copy_inputs(self):
        """Fork the main stream into the copy stream for the input copies.

        The caller writes the request into the static buffers inside the ``with`` body; the
        fork/join is exactly the idom all three backends use::

            runner = GraphRunner(...)
            with runner.copy_inputs() as copy_stream:       # copy_stream.wait_stream(main)
                static["images"][i].copy_(img)
                ...
            runner.wait_inputs()                           # main waits ev_in
            graph.replay()

        Both events exist so the copies (and, for smolvla, the noise sample) overlap the
        caller's main-stream work without ever reordering against the replay.
        """
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self.ev_in = torch.cuda.Event()
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(main)  # copy sources were produced on main
            yield self.copy_stream
            assert self.ev_in is not None
            self.ev_in.record(self.copy_stream)

    def wait_inputs(self) -> None:
        """Make the main stream wait for the copies issued by :meth:`copy_inputs`."""
        assert self.ev_in is not None, "copy_inputs() must run before wait_inputs()"
        torch.cuda.current_stream(self.device).wait_event(self.ev_in)


def _fmt_key(key: Hashable) -> str:
    if isinstance(key, tuple):
        return "(" + ", ".join(str(k) for k in key) + ")"
    return str(key)
