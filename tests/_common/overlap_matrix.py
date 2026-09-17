"""The 4-mode overlap matrix shared by the pi05 and smolvla checks.

Modes (every row is requested with ``overlap=True``):

  1. ``eager``        -- no CUDA graph, plain kernels
  2. ``eager+graph``  -- CUDA graph, plain kernels
  3. ``fused``        -- no CUDA graph, fused (Triton) tiers on
  4. ``fused+graph``  -- CUDA graph, fused (Triton) tiers on

Overlap is a *model-level execution mode*: the model interleaves its step-0 expert layers with its
LLM prefill, and a CUDA graph only captures whichever branch is active. Every row must therefore
be bit-exact against the same row with overlap off, in **both** graph and non-graph modes:

* graph rows: the engine drives the model flag (``overlap`` stays gated on the graph path);
* non-graph rows: the model flag is driven directly, which is what a deployment without CUDA
  graphs would do -- the engines keep it off there because the eager-mode gain is
  platform-dependent and has not been broadly tested, not because the implementation is
  graph-specific.

Each ``(mode, overlap)`` pair is its own process -- a captured graph owns a private CUDA pool --
so the two halves of a row are compared by the driver, on the output vectors the children send
back in their rows.

``quick`` covers the production tier's graph row (``fused+graph``); ``full`` adds the eager rows and
the non-fused tier, which are the control for the intra-tier overlap-vs-sequential comparison.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cli import (
    ModelSpec,
    add_engine_arguments,
    check_description,
    job_keys,
    missing_environment,
    print_job_list,
)
from .level import Level, jobs_for_level
from .process import run_jobs
from .report import Column, finish, print_table, skip
from .rows import RowResult, RowStatus, emit_row

__all__ = ["MatrixJob", "MatrixMode", "MODES", "matrix_jobs", "overlap_matrix_check"]

NOTE = (
    "NOTE: overlap is a model-level execution mode; a graph merely captures the active branch.\n"
    "      Every row must therefore be bit-exact against the same row with overlap off -- in\n"
    "      graph AND non-graph modes."
)

COLUMNS = (
    Column("mode", "key", "<"),
    Column("graph", "graph"),
    Column("overlap", "overlap_engaged"),
    Column("graphs", "graphs"),
    Column("run", "run"),
    Column("d(ovl, no-ovl)", "max_abs_diff_vs_sequential", "{:.3e}"),
    Column("verdict", "verdict", "<"),
    Column("result", "status", "<"),
)


@dataclass(frozen=True)
class MatrixMode:
    """One execution mode of the matrix."""

    key: str
    graph: bool
    quick: bool


@dataclass(frozen=True)
class MatrixJob:
    """One process: a mode with overlap either requested or off (the row's reference)."""

    mode: MatrixMode
    overlap: bool

    @property
    def quick(self) -> bool:
        return self.mode.quick

    @property
    def key(self) -> str:
        return f"{self.mode.key} overlap={'on' if self.overlap else 'off'}"


MODES: tuple[MatrixMode, ...] = (
    MatrixMode("eager", graph=False, quick=False),
    MatrixMode("eager+graph", graph=True, quick=False),
    MatrixMode("fused", graph=False, quick=False),
    # the production tier (fused kernels, captured) and the only row the quick level needs
    MatrixMode("fused+graph", graph=True, quick=True),
)


def matrix_jobs(level: Level) -> list[MatrixJob]:
    """The jobs of ``level``, ordered as ``(overlap on, overlap off)`` pairs per mode."""
    jobs = [MatrixJob(mode, overlap) for mode in MODES for overlap in (True, False)]
    return jobs_for_level(jobs, level)


def _count_graphs(obj: Any) -> int:
    """Number of ``CUDAGraph`` objects reachable from a registry entry."""
    import torch

    if isinstance(obj, torch.cuda.CUDAGraph):
        return 1
    if isinstance(obj, dict):
        return sum(_count_graphs(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_graphs(value) for value in obj)
    return 0


def _run_job(spec: ModelSpec, job: MatrixJob, *, checkpoint: str, device: str) -> RowResult:
    """Child side: build the engine, run one request, report metrics and the raw output."""
    import numpy as np
    import torch

    from tybok.registry import create_engine

    identity = {"mode": job.mode.key, "graph": int(job.mode.graph), "overlap": int(job.overlap)}
    flags = spec.fused_flags if job.mode.key.startswith("fused") else ()
    metrics: dict[str, Any] = {
        "graph": int(job.mode.graph),
        "overlap_requested": int(job.overlap),
    }

    try:
        engine = create_engine(
            checkpoint,
            model_type=spec.key,
            device=device,
            graph=job.mode.graph,
            overlap=job.overlap,
            warmup=True,
            **{flag: True for flag in flags},
        )

        # ``PolicyEngine`` does not declare ``policy`` (every backend sets it), hence getattr.
        model = getattr(getattr(engine, "policy", None), "model", None)
        if job.overlap and not job.mode.graph and model is not None:
            # No graph to fall back on: drive the model's own branch directly, which is what an
            # eager deployment would do once the flag is not graph-gated. All three backends
            # expose the same model-level flag (``model.overlap``).
            model.overlap = True
        runner = getattr(engine, "_graph_runner", None)
        metrics["overlap_engaged"] = int(
            bool(getattr(engine, "overlap", False)) or bool(getattr(model, "overlap", False))
        )

        metrics["graphs"] = sum(_count_graphs(entry) for entry in runner.entries.values()) if runner is not None else 0

        spec_dict = engine.describe()
        frame = engine._profile_frame(seed=1)
        noise = torch.randn(
            1,
            spec_dict["chunk_size"],
            spec_dict["action_dim"],
            generator=torch.Generator().manual_seed(123),
        )

        output = np.asarray(engine.predict_action_chunk(frame, noise=noise), dtype=np.float64)
    except Exception as error:  # noqa: BLE001 - a row that cannot run is a result, not a crash
        return RowResult(
            key=job.key,
            status=RowStatus.FAIL,
            detail=f"error: {type(error).__name__}: {error}",
            job=identity,
            metrics=metrics,
        )

    return RowResult(
        key=job.key,
        status=RowStatus.PASS,
        job=identity,
        metrics=metrics,
        values=[float(value) for value in output.ravel()],
    )


def _find(rows: Sequence[RowResult], *, mode: str, overlap: bool) -> RowResult | None:
    wanted = {"mode": mode, "overlap": int(overlap)}
    for row in rows:
        if all(row.job.get(field) == value for field, value in wanted.items()):
            return row
    return None


def _judge(mode: MatrixMode, rows: Sequence[RowResult]) -> RowResult:
    """Compare a mode's overlap-on and overlap-off processes, as the old matrix did."""
    import numpy as np

    active = _find(rows, mode=mode.key, overlap=True)
    reference = _find(rows, mode=mode.key, overlap=False)
    metrics: dict[str, Any] = dict(active.metrics) if active is not None else {}
    metrics["graph"] = int(mode.graph)

    if active is None or not active.is_pass or active.values is None:
        detail = active.detail if active is not None else "row missing"
        return RowResult(mode.key, RowStatus.FAIL, f"RUN FAILED ({detail})", metrics=metrics)
    if reference is None or not reference.is_pass or reference.values is None:
        detail = reference.detail if reference is not None else "row missing"
        return RowResult(mode.key, RowStatus.FAIL, f"REFERENCE FAILED ({detail})", metrics=metrics)

    metrics["run"] = "ok"
    diff = float(
        np.abs(np.asarray(active.values, dtype=np.float64) - np.asarray(reference.values, dtype=np.float64)).max()
    )

    metrics["max_abs_diff_vs_sequential"] = diff
    engaged = bool(metrics.get("overlap_engaged"))

    if not engaged:
        # Nothing to verify in this mode: the engine deliberately keeps overlap off here.
        verdict = "overlap off (not enabled for this tier)"
        return RowResult(mode.key, RowStatus.PASS, verdict, metrics={**metrics, "verdict": verdict})
    if diff == 0.0:
        verdict = "supported (graph)" if mode.graph else "supported (model-level, no graph)"
        return RowResult(mode.key, RowStatus.PASS, verdict, metrics={**metrics, "verdict": verdict})
    return RowResult(
        mode.key,
        RowStatus.FAIL,
        f"not bit-exact ({diff:.3e})",
        metrics={**metrics, "verdict": "NOT BIT-EXACT"},
    )


def _parse_args(argv: Sequence[str] | None, spec: ModelSpec, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(description))
    add_engine_arguments(parser, spec)
    return parser.parse_args(argv)


def overlap_matrix_check(
    spec: ModelSpec,
    script: Path,
    description: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Entry point of a per-model overlap matrix check (driver and child in one)."""
    args = _parse_args(argv, spec, description)
    jobs = matrix_jobs(args.level)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        job = jobs[args.job_index]
        reason = missing_environment(args.device, args.checkpoint)
        if reason is not None:
            identity = {"mode": job.mode.key, "graph": int(job.mode.graph), "overlap": int(job.overlap)}
            emit_row(RowResult(job.key, RowStatus.SKIP, reason, job=identity))
            return 0
        emit_row(_run_job(spec, job, checkpoint=args.checkpoint, device=args.device))
        return 0

    reason = missing_environment(args.device, args.checkpoint)
    if reason is not None:
        return skip(reason, require_gpu=args.require_gpu)

    print(f"\n=== {spec.key} overlap matrix ({args.level.value} level) ===")
    print(f"checkpoint: {args.checkpoint}")
    rows = run_jobs(
        script,
        job_keys(jobs),
        ["--level", args.level.value, "--device", args.device, "--checkpoint", args.checkpoint],
    )

    verdicts = [_judge(mode, rows) for mode in MODES if mode in {job.mode for job in jobs}]
    print()
    print_table(COLUMNS, verdicts)
    print()
    print(NOTE)
    return finish(verdicts, pass_message=f"{spec.key}: overlap supported in every row")
