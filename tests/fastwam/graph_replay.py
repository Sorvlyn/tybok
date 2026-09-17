"""fastwam CUDA-graph replay check: the sequential core and the overlapped core.

Validates the production runner (``tybok/policies/fastwam/models/graph_runner.py``) against the
eager path:

  * the first replay is bit-exact vs eager;
  * idempotency over many replays -- the historical replay-drift bug: an input tensor the graph
    reads was referenced only by a local closure, so the allocator reused its memory and
    replay #2 onwards read corrupted data (this is the keepalive invariant);
  * correctness with *different* inputs across replays (the runner re-fills its static buffers);
  * stability with eager calls interleaved between replays;
  * the eager fused branch is bit-exact against the sequential path, so one eager reference
    serves both cores.

Both capture paths run because ``_denoise_core`` has two of them: ``sequential`` and ``overlap``
(the multi-stream prefill x step-0 fork/join). The kernel tier is the production one
(:data:`tests.fastwam.spec.PRODUCTION_TIER`); tier x graph-mode combinations are
``tests/fastwam/overlap_matrix.py``'s job.

Checks are bit-exact by default -- a single shot per comparison. ``--attempts`` /
``--reference-attempts`` re-introduce retries and are for *diagnosing* a suspected flake only; a
gate that tolerates a rare mismatch by default is exactly what hid the 2026-09-14 kernel bug for
so long (see the history below).

Jobs (``--list-jobs``): ``sequential core``, ``overlap core``.

Why this check exists: the fused ``vdit_attn_cross`` kernel used to be nondeterministic for
identical inputs (~0.1-0.7% of calls) with **no CUDA graph involved** -- ``C_PHASE_2`` calls
``fwam_fp8_gemm_body`` twice back to back with no barrier between the calls, so one thread could
prefetch into memory the previous call's epilogue was still staging. The barrier is in place now;
this check locks it in (many replays, with alternate inputs, must stay bit-exact).

Usage::

    cd TyBoK
    python tests/fastwam/graph_replay.py                    # quick, both cores
    python tests/fastwam/graph_replay.py --level full --replays 200
    python tests/fastwam/graph_replay.py --list-jobs
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Allow ``python tests/fastwam/graph_replay.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common import (  # noqa: E402 - after the path bootstrap
    Column,
    Level,
    RowResult,
    RowStatus,
    add_engine_arguments,
    check_description,
    emit_row,
    finish,
    job_keys,
    jobs_for_level,
    missing_environment,
    print_job_list,
    print_table,
    run_jobs,
    skip,
)
from tests.fastwam.spec import FASTWAM, PRODUCTION_TIER  # noqa: E402 - after the path bootstrap

__all__ = ["main"]

TASK = "pick up the cup"
QUICK_REPLAYS = 5
FULL_REPLAYS = 20
DEFAULT_STEPS = 10

COLUMNS = (
    Column("core", "key", "<"),
    Column("replays", "replays"),
    Column("graphs", "graphs"),
    Column("first replay", "first_replay", "{:.3e}"),
    Column("idempotency", "idempotency", "{:.3e}"),
    Column("alternating input", "alternating_input", "{:.3e}"),
    Column("eager ovl vs seq", "eager_overlap_vs_sequential", "{:.3e}"),
    Column("interleaved graph", "interleaved_graph", "{:.3e}"),
    Column("nan rounds", "nan_rounds"),
    Column("retries", "retries"),
    Column("result", "status", "<"),
)


class Core(str, Enum):
    """Which capture path of ``_denoise_core`` to exercise."""

    SEQUENTIAL = "sequential"
    OVERLAP = "overlap"


@dataclass(frozen=True)
class ReplayJob:
    """One captured core, replayed and compared against the eager path."""

    core: Core

    @property
    def quick(self) -> bool:
        return True

    @property
    def key(self) -> str:
        return f"{self.core.value} core"


JOBS: tuple[ReplayJob, ...] = (ReplayJob(Core.SEQUENTIAL), ReplayJob(Core.OVERLAP))


@dataclass
class Measurer:
    """Comparison primitives, with optional retries for diagnosing a suspected flake."""

    attempts: int = 1
    reference_attempts: int = 1
    retries: int = 0
    calls: int = 0

    def reference(self, run: Callable[[], Any]) -> Any:
        """Majority value of ``reference_attempts`` eager runs, so one flaky run cannot be it."""
        import numpy as np

        results = [np.array(run(), copy=True) for _ in range(self.reference_attempts)]
        for index, candidate in enumerate(results):
            for other in results[index + 1 :]:
                if candidate.shape == other.shape and np.array_equal(candidate, other):
                    return candidate
        return results[0]  # every run differed (very unlikely); the phases report it

    def deviation(self, run: Callable[[], Any], reference: Any) -> float:
        """Min deviation over ``attempts`` attempts; ``0.0`` iff one attempt is bit-exact.

        A correct-but-flaky path is retried away; a systematically wrong path fails on every
        attempt, so nothing is hidden.
        """
        best: float | None = None
        for attempt in range(self.attempts):
            self.calls += 1
            value = _max_abs_diff(run(), reference)
            if value == 0.0:
                if attempt:
                    self.retries += 1
                return 0.0
            best = value if best is None else min(best, value)
        self.retries += 1
        # ``attempts >= 1``, so ``best`` was set by the loop (``None`` would mean no attempt ran)
        return best if best is not None else float("nan")


def _max_abs_diff(actual: Any, expected: Any) -> float:
    """``max|actual - expected|``; a NaN is never swallowed."""
    import torch

    difference = (torch.as_tensor(actual).float() - torch.as_tensor(expected).float()).abs().max()
    return float(difference)


def _nan_safe_max(current: float, new: float) -> float:
    """NaN-safe ``max``: a NaN must not be discarded by a later comparison."""
    if math.isnan(new):
        return new
    if math.isnan(current):
        return current
    return max(current, new)


def _make_frame(engine: Any, spec: dict[str, Any], seed: int) -> Any:
    import torch

    generator = torch.Generator().manual_seed(seed)
    cameras = [camera.split("observation.images.")[-1] for camera in spec["cameras"]]
    images = {camera: torch.rand(3, spec["resize"][1], spec["resize"][0], generator=generator) for camera in cameras}
    state = torch.rand(engine.config.proprio_dim or 8, generator=generator)
    return engine.make_frame(images, state, TASK)


def _make_noise(engine: Any, seed: int) -> Any:
    import torch

    return torch.randn(
        1,
        engine.config.action_horizon,
        engine.config.action_dim,
        generator=torch.Generator().manual_seed(seed),
    )


def _run_core(
    job: ReplayJob,
    *,
    checkpoint: str,
    device: str,
    text_encoder_device: str,
    steps: int,
    replays: int,
    measurer: Measurer,
) -> RowResult:
    """Child side: capture one core, run every phase, report the deviations."""
    import torch

    from tybok.policies.fastwam.engine import FastWAMEngine
    from tybok.policies.fastwam.models.graph_runner import FastWAMGraphRunner

    torch.manual_seed(0)
    # ``overlap=True`` so the eager reference exercises the fused two-stream branch; the runner
    # then captures whichever branch the job selects. Both are bit-exact against the sequential
    # path, so one eager reference serves both cores.
    engine = FastWAMEngine(
        checkpoint_dir=checkpoint,
        device=device,
        warmup=True,
        text_encoder_device=text_encoder_device,
        num_steps=steps,
        overlap=True,
        **PRODUCTION_TIER,
    )

    # ``PolicyEngine`` does not declare ``policy`` (every backend sets it), hence getattr.
    model = getattr(getattr(engine, "policy", None), "model", None)
    spec = engine.describe()

    frame_first = _make_frame(engine, spec, seed=1)
    frame_alternate = _make_frame(engine, spec, seed=2)
    noise_first = _make_noise(engine, seed=11)
    noise_alternate = _make_noise(engine, seed=22)

    model.overlap = True
    reference_first = measurer.reference(lambda: engine.predict_action_chunk(frame_first, noise=noise_first))

    reference_alternate = measurer.reference(
        lambda: engine.predict_action_chunk(frame_alternate, noise=noise_alternate)
    )

    runner = FastWAMGraphRunner(model, device, overlap=job.core is Core.OVERLAP)
    model._graph_runner = runner

    metrics: dict[str, Any] = {"replays": replays, "graphs": runner.num_graphs}
    metrics["first_replay"] = measurer.deviation(
        lambda: engine.predict_action_chunk(frame_first, noise=noise_first), reference_first
    )

    idempotency = 0.0
    alternating = 0.0
    for _ in range(replays):
        idempotency = _nan_safe_max(
            idempotency,
            measurer.deviation(
                lambda: engine.predict_action_chunk(frame_first, noise=noise_first),
                reference_first,
            ),
        )

        alternating = _nan_safe_max(
            alternating,
            measurer.deviation(
                lambda: engine.predict_action_chunk(frame_alternate, noise=noise_alternate),
                reference_alternate,
            ),
        )

    metrics["idempotency"] = idempotency
    metrics["alternating_input"] = alternating

    # Interleave eager work (the graph runner detached): what the eager reference becomes while a
    # graph runner is attached, and the replay that follows it.
    model._graph_runner = None
    model.overlap = False
    metrics["eager_overlap_vs_sequential"] = measurer.deviation(
        lambda: engine.predict_action_chunk(frame_first, noise=noise_first), reference_first
    )

    interleaved_eager = 0.0
    interleaved_graph = 0.0
    nan_rounds: list[int] = []
    for round_index in range(replays):
        model._graph_runner = None
        interleaved_eager = _nan_safe_max(
            interleaved_eager,
            measurer.deviation(
                lambda: engine.predict_action_chunk(frame_first, noise=noise_first),
                reference_first,
            ),
        )

        model._graph_runner = runner
        deviation = measurer.deviation(
            lambda: engine.predict_action_chunk(frame_first, noise=noise_first), reference_first
        )

        if math.isnan(deviation):
            nan_rounds.append(round_index)
        else:
            interleaved_graph = _nan_safe_max(interleaved_graph, deviation)
    metrics["interleaved_eager"] = interleaved_eager
    metrics["interleaved_graph"] = interleaved_graph
    metrics["nan_rounds"] = len(nan_rounds)
    metrics["retries"] = measurer.retries
    metrics["model_calls"] = measurer.calls

    failed = [
        name
        for name in (
            "first_replay",
            "idempotency",
            "alternating_input",
            "eager_overlap_vs_sequential",
            "interleaved_eager",
            "interleaved_graph",
        )
        if metrics[name] != 0.0
    ]
    if nan_rounds:
        failed.append(f"nan in rounds {nan_rounds}")
    status = RowStatus.FAIL if failed else RowStatus.PASS
    detail = f"{', '.join(failed)} differ from eager" if failed else ""
    return RowResult(job.key, status, detail, job={"core": job.core.value}, metrics=metrics)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(__doc__))
    add_engine_arguments(parser, FASTWAM)
    parser.add_argument("--text-encoder-device", default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="denoising steps")
    parser.add_argument(
        "--replays",
        type=int,
        default=None,
        help=f"replays per phase (default: {QUICK_REPLAYS} quick / {FULL_REPLAYS} full)",
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="attempts per comparison; 1 = single shot, bit-exact. Raise it only to diagnose a "
        "suspected flake (see the module docstring)",
    )

    parser.add_argument(
        "--reference-attempts",
        type=int,
        default=1,
        help="eager runs per reference, majority vote (default: 1)",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay check (driver), or one core of it (``--job-index``)."""
    args = _parse_args(argv)
    jobs = jobs_for_level(JOBS, args.level)
    replays = args.replays or (QUICK_REPLAYS if args.level is Level.QUICK else FULL_REPLAYS)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        job = jobs[args.job_index]
        reason = missing_environment(args.device, args.checkpoint)
        if reason is not None:
            emit_row(RowResult(job.key, RowStatus.SKIP, reason, job={"core": job.core.value}))
            return 0
        measurer = Measurer(attempts=args.attempts, reference_attempts=args.reference_attempts)
        emit_row(
            _run_core(
                job,
                checkpoint=args.checkpoint,
                device=args.device,
                text_encoder_device=args.text_encoder_device,
                steps=args.steps,
                replays=replays,
                measurer=measurer,
            )
        )
        return 0

    reason = missing_environment(args.device, args.checkpoint)
    if reason is not None:
        return skip(reason, require_gpu=args.require_gpu)

    print(f"\n=== fastwam graph replay ({args.level.value} level, {replays} replays) ===")
    print(f"checkpoint: {args.checkpoint}")
    rows = run_jobs(
        Path(__file__),
        job_keys(jobs),
        [
            "--level",
            args.level.value,
            "--device",
            args.device,
            "--checkpoint",
            args.checkpoint,
            "--text-encoder-device",
            args.text_encoder_device,
            "--steps",
            str(args.steps),
            "--replays",
            str(replays),
            "--attempts",
            str(args.attempts),
            "--reference-attempts",
            str(args.reference_attempts),
        ],
    )

    print()
    print_table(COLUMNS, rows)
    print()
    print("NOTE: every phase must be exactly 0.000e+00 -- the keepalive invariant (every tensor")
    print("      the captured graph reads stays alive for the graph's whole lifetime).")
    return finish(rows, pass_message=f"fastwam: both cores bit-exact over {replays} replays")


if __name__ == "__main__":
    sys.exit(main())
