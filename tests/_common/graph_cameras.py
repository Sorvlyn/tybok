"""The camera-count shape-bucket gate shared by the fastwam / pi05 / smolvla checks.

The requirement behind the unified graph plumbing (``tybok.graph``): the engines key their graphs
on the number of model-level image inputs, so a gateway that sends 2 or 3 cameras can pre-capture
both shapes at startup and never capture on the request path.

* ``smolvla`` / ``pi05``: one ViT pass per camera slot, so the prefix length -- and therefore the
  captured static buffers -- changes with the count: genuinely different graphs, one per count.
* ``fastwam``: the camera images are concatenated into **one** frame by the preprocessor and the
  VAE runs outside the captured core, so there is one graph shape for 2 and 3 cameras alike;
  ``--graph-cameras`` is accepted and logged as a no-op. The row asserts it is accepted and that
  the engine still runs.

Method: per model, one process runs every clamped camera count with ``graph=False`` (the eager
reference) and another with ``graph=True, graph_cameras=(2, 3)``; each count's output must match
bit-exactly. One engine per process, because two pi05 instances do not fit in 16 GB together.

The counts are clamped to what the checkpoint exposes, which is the clamping the engines document
(`pi05_libero` has 2 camera keys, so it exercises 2 and not 3).

``quick`` runs the same two jobs per model as ``full`` -- the check is two engine builds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

__all__ = ["CameraConfig", "CameraJob", "camera_jobs", "graph_cameras_check"]

#: The camera counts a gateway may send; each is pre-captured at startup.
REQUESTED_CAMERAS = (2, 3)

COLUMNS = (
    Column("model", "model", "<"),
    Column("pre-captured", "precaptured", "<"),
    Column("counts", "counts", "<"),
    Column("graphs", "graphs"),
    Column("max|graph-eager|", "max_abs_diff_vs_eager", "{:.3e}"),
    Column("result", "status", "<"),
)


@dataclass(frozen=True)
class CameraConfig:
    """A model plus the engine options this check builds it with."""

    spec: ModelSpec
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraJob:
    """One process: every clamped count with graphs off (reference) or on."""

    graph: bool

    @property
    def quick(self) -> bool:
        return True

    @property
    def key(self) -> str:
        return "graph" if self.graph else "eager"


def camera_jobs(level: Level) -> list[CameraJob]:
    """The jobs of ``level`` (eager reference first, then the graph run)."""
    return jobs_for_level([CameraJob(False), CameraJob(True)], level)


def _run_job(config: CameraConfig, job: CameraJob, *, device: str, checkpoint: str) -> RowResult:
    """Child side: build the engine, run every clamped camera count, send back the outputs."""
    import numpy as np
    import torch

    from tybok.registry import create_engine

    identity = {"model": config.spec.key, "graph": int(job.graph)}
    metrics: dict[str, Any] = {"model": config.spec.key}
    try:
        engine = create_engine(
            checkpoint,
            model_type=config.spec.key,
            device=device,
            graph=job.graph,
            overlap=True,
            graph_cameras=REQUESTED_CAMERAS,
            warmup=True,
            **config.extra_kwargs,
        )

        spec = engine.describe()
        max_cameras = len(spec["cameras"])
        # ``_graph_runner`` is None for fastwam when graphs are off (an empty registry for
        # smolvla/pi05); both are falsy, so the engines' own ``if self._graph_runner:`` reads the
        # same. The keys are what the engine really pre-captured.
        runner = getattr(engine, "_graph_runner", None)
        counts = [count for count in REQUESTED_CAMERAS if count <= max_cameras] or [max_cameras]
        # The keys are what the engine really pre-captured. Every backend's key starts with the
        # number of model-level image inputs (that is the convention ``tybok.graph`` establishes),
        # so the table shows that element and keeps the full keys for the record -- fastwam's key
        # carries shapes and dtypes and would swamp the row.
        # ``runner.keys()``: ``GraphRunner`` defines ``keys()`` and ``__len__`` but not ``__iter__``
        keys = [tuple(key) if isinstance(key, tuple) else (key,) for key in runner.keys()] if runner else []
        metrics.update(
            max_cameras=max_cameras,
            counts=", ".join(str(count) for count in counts),
            precaptured=", ".join(str(key[0]) for key in keys) if keys else "-",
            graphs=len(keys),
            graph_keys=[str(list(key)) for key in keys],
        )

        values: list[float] = []
        for count in counts:
            frame = engine._profile_frame(cameras=count, seed=1)
            noise = torch.randn(
                1,
                spec["chunk_size"],
                spec["action_dim"],
                generator=torch.Generator().manual_seed(123),
            )

            output = np.asarray(engine.predict_action_chunk(frame, noise=noise), dtype=np.float64)
            values.extend(float(value) for value in output.ravel())
    except Exception as error:  # noqa: BLE001 - a count that cannot run is a result, not a crash
        return RowResult(
            job.key,
            RowStatus.FAIL,
            f"error: {type(error).__name__}: {error}",
            job=identity,
            metrics=metrics,
        )

    return RowResult(job.key, RowStatus.PASS, job=identity, metrics=metrics, values=values)


def _split_by_count(row: RowResult, counts: Sequence[int]) -> list[list[float]]:
    """Split a row's flat output into one vector per camera count (equal shapes, see ``_run_job``)."""
    if row.values is None or not counts:
        return []
    per_count = len(row.values) // len(counts)
    return [list(row.values[index * per_count : (index + 1) * per_count]) for index in range(len(counts))]


def _judge(config: CameraConfig, rows: Sequence[RowResult]) -> RowResult:
    """Compare the graph run against the eager reference, count by count."""
    import numpy as np

    eager = _find(rows, graph=False)
    graph = _find(rows, graph=True)
    metrics: dict[str, Any] = {"model": config.spec.key}
    source = graph if graph is not None and graph.is_pass else eager
    if source is not None:
        metrics.update(source.metrics)

    for name, row in (("eager", eager), ("graph", graph)):
        if row is None or not row.is_pass:
            detail = row.detail if row is not None else "row missing"
            return RowResult(config.spec.key, RowStatus.FAIL, f"{name} failed ({detail})", metrics=metrics)
    if eager is None or graph is None or eager.values is None or graph.values is None:
        return RowResult(config.spec.key, RowStatus.FAIL, "missing output", metrics=metrics)

    counts = [int(count) for count in str(metrics["counts"]).split(", ") if count]
    eager_vectors, graph_vectors = _split_by_count(eager, counts), _split_by_count(graph, counts)
    worst = 0.0
    for count, expected, actual in zip(counts, eager_vectors, graph_vectors):
        difference = float(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)).max())

        if difference != 0.0:
            metrics["max_abs_diff_vs_eager"] = difference
            return RowResult(
                config.spec.key,
                RowStatus.FAIL,
                f"{count} cameras differ by {difference:.3e}",
                metrics=metrics,
            )

        worst = max(worst, difference)
    metrics["max_abs_diff_vs_eager"] = worst
    return RowResult(config.spec.key, RowStatus.PASS, metrics=metrics)


def _find(rows: Sequence[RowResult], *, graph: bool) -> RowResult | None:
    for row in rows:
        if row.job.get("graph") == int(graph):
            return row
    return None


def _parse_args(argv: Sequence[str] | None, config: CameraConfig, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(description))
    add_engine_arguments(parser, config.spec)
    return parser.parse_args(argv)


def graph_cameras_check(
    config: CameraConfig,
    script: Path,
    description: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Entry point of a per-model camera-count check (driver and child in one)."""
    args = _parse_args(argv, config, description)
    jobs = camera_jobs(args.level)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        job = jobs[args.job_index]
        reason = missing_environment(args.device, args.checkpoint)
        identity = {"model": config.spec.key, "graph": int(job.graph)}
        if reason is not None:
            emit_row(RowResult(job.key, RowStatus.SKIP, reason, job=identity))
            return 0
        emit_row(_run_job(config, job, device=args.device, checkpoint=args.checkpoint))
        return 0

    reason = missing_environment(args.device, args.checkpoint)
    if reason is not None:
        return skip(reason, require_gpu=args.require_gpu)

    print(f"\n=== {config.spec.key} graph cameras ({args.level.value} level) ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"requested camera counts: {REQUESTED_CAMERAS} (clamped to the checkpoint)")
    rows = run_jobs(
        script,
        job_keys(jobs),
        [
            "--level",
            args.level.value,
            "--device",
            args.device,
            "--checkpoint",
            args.checkpoint,
        ],
    )

    print()
    verdicts = [_judge(config, rows)]
    print_table(COLUMNS, verdicts)
    print()
    print("NOTE: the graphs are keyed on the number of model-level image inputs (the 'pre-captured'")
    print("      column), so a gateway that sends 2 or 3 cameras never captures on the request path.")
    return finish(verdicts, pass_message=f"{config.spec.key}: every camera count bit-exact")
