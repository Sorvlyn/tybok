"""The ``--compile`` mode gate shared by the fastwam / pi05 / smolvla checks.

Modes (all requesting ``compile_model=True``):

  1. ``eager+compile``                -- torch.compile, no CUDA graph
  2. ``eager+compile+graph``          -- torch.compile + CUDA graph  (the deployment combination)
  3. ``eager+compile+graph+overlap``  -- torch.compile + CUDA graph + the overlap branch

Per row it reports whether the engine was constructed at all, whether it *kept*
``compile_model`` and ``overlap`` (the engines force them off in combinations they do not
support: ``--compile`` disables ``--overlap``), which regions really became dynamo
``OptimizedModule``s, and whether a request runs. The region names are not invented here: each
engine declares the attribute paths it wrapped (``eng.compiled_regions``), this check verifies each
one actually carries a dynamo wrapper, and reports it under that same name (``embed_prefix`` /
``vlm_with_expert.forward`` / ``paligemma_with_expert.forward`` / ``mot.prefill_video_cache`` /
``denoise_step``). A declared region without a wrapper is a failure: the wiring silently did not
take effect, which is exactly the "two paths, two rule sets" class of bug this suite guards.

Each job runs in its own process (one engine per process keeps peak memory at roughly one engine),
and a row that cannot run is a *result*, not a crash. All three backends support ``--compile``
now, so a row that does not run is a regression.

``quick`` runs the deployment combination (``eager+compile+graph``) only: compiling is the
expensive part of this check and the other two modes are combinations that exist to document what
they do, not to be deployed.
"""

from __future__ import annotations

import argparse
import time
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

__all__ = ["CompileConfig", "CompileJob", "compile_jobs", "compile_modes_check"]

COLUMNS = (
    Column("model", "model", "<"),
    Column("mode", "key", "<"),
    Column("compile", "compile_kept", "^"),
    Column("dynamo", "compiled_module", "^"),
    Column("graph", "graph_enabled", "^"),
    Column("overlap", "overlap_kept", "^"),
    Column("ms", "ms", "{:.1f}"),
    Column("compiled regions", "regions", "<"),
    Column("result", "status", "<"),
)


@dataclass(frozen=True)
class CompileConfig:
    """A model plus the engine options this check builds it with."""

    spec: ModelSpec
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileMode:
    """One combination of compile / graph / overlap."""

    key: str
    graph: bool
    overlap: bool
    quick: bool


@dataclass(frozen=True)
class CompileJob:
    """One process: a mode built once and timed."""

    mode: CompileMode

    @property
    def quick(self) -> bool:
        return self.mode.quick

    @property
    def key(self) -> str:
        return self.mode.key


MODES: tuple[CompileMode, ...] = (
    CompileMode("eager+compile", graph=False, overlap=False, quick=False),
    # the deployment combination: compiled regions inside a captured graph
    CompileMode("eager+compile+graph", graph=True, overlap=False, quick=True),
    CompileMode("eager+compile+graph+overlap", graph=True, overlap=True, quick=False),
)


def compile_jobs(level: Level) -> list[CompileJob]:
    """The jobs of ``level`` (the modes in declaration order)."""
    return jobs_for_level([CompileJob(mode) for mode in MODES], level)


def _is_compiled(callable_: Any) -> bool:
    """Whether dynamo wrapped ``callable_`` (on a bound method ``_orig_mod`` is not always set)."""
    if callable_ is None:
        return False
    return bool(
        getattr(callable_, "_orig_mod", None) is not None
        or getattr(callable_, "_torch_dynamo_orig_callable", None) is not None
        or getattr(callable_, "_torchdynamo_orig_callable", None) is not None
        or "_dynamo" in type(callable_).__module__
        or "OptimizedModule" in type(callable_).__name__
    )


def _resolve(model: Any, path: str) -> Any:
    """Follow an engine-declared attribute path (``vlm_with_expert.forward``)."""
    resolved = model
    for part in path.split("."):
        resolved = getattr(resolved, part, None)
        if resolved is None:
            break
    return resolved


def _run_job(config: CompileConfig, job: CompileJob, *, device: str, checkpoint: str) -> RowResult:
    """Child side: build the combination, verify the regions took effect, time one request."""
    import numpy as np
    import torch

    from tybok.registry import create_engine

    identity = {"model": config.spec.key, "mode": job.mode.key}
    metrics: dict[str, Any] = {"model": config.spec.key}
    try:
        engine = create_engine(
            checkpoint,
            model_type=config.spec.key,
            device=device,
            compile_model=True,
            graph=job.mode.graph,
            overlap=job.mode.overlap,
            warmup=True,
            **config.extra_kwargs,
        )

        spec = engine.describe()
        model = getattr(getattr(engine, "policy", None), "model", None)

        # Region names come from the engine (the attribute paths it wrapped); each one is
        # verified to carry a dynamo wrapper, so a region that was declared but did not take
        # effect shows up as missing instead of being trusted.
        declared = list(getattr(engine, "compiled_regions", []))
        resolved = {path: _resolve(model, path) for path in declared}
        regions = [path for path, target in resolved.items() if _is_compiled(target)]
        missing = [path for path, target in resolved.items() if not _is_compiled(target)]
        sample_actions = getattr(model, "sample_actions", None)
        # ``len(runner)``: the fastwam runner and ``tybok.graph.GraphRunner`` both define it,
        # whereas ``num_graphs`` exists on the fastwam one only
        graph_runner = getattr(engine, "_graph_runner", None)
        metrics.update(
            compile_kept=bool(getattr(engine, "compile_model", False)),
            compiled_module=_is_compiled(sample_actions) or bool(regions),
            graph_enabled=bool(getattr(engine, "graph_enabled", False)),
            # what the engine ended up running with, not what was requested
            overlap_kept=bool(getattr(engine, "overlap", False)),
            regions=", ".join(regions) if regions else "-",
            missing_regions=", ".join(missing),
            graphs=(len(graph_runner) if graph_runner is not None else 0),
        )

        frame = engine._profile_frame(seed=1)
        noise = torch.randn(
            1,
            spec["chunk_size"],
            spec["action_dim"],
            generator=torch.Generator().manual_seed(123),
        )

        engine.predict_action_chunk(frame, noise=noise)  # warm: fills caches, replays once
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        output: np.ndarray | None = None
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            output = np.asarray(engine.predict_action_chunk(frame, noise=noise), dtype=np.float64)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1e3)
        # ``range(5)`` always binds it; the assert says so for the type checker.
        assert output is not None
        metrics["ms"] = float(sorted(timings)[len(timings) // 2])
        metrics["finite"] = bool(np.isfinite(output).all())
    except Exception as error:  # noqa: BLE001 - an unrunnable row is a result, not a crash
        return RowResult(
            job.key,
            RowStatus.FAIL,
            f"error: {type(error).__name__}: {error}",
            job=identity,
            metrics=metrics,
        )

    if not metrics["compiled_module"]:
        return RowResult(job.key, RowStatus.FAIL, "no compiled region took effect", job=identity, metrics=metrics)

    if metrics["missing_regions"]:
        return RowResult(
            job.key,
            RowStatus.FAIL,
            f"declared but not compiled: {metrics['missing_regions']}",
            job=identity,
            metrics=metrics,
        )

    if not metrics["finite"]:
        return RowResult(job.key, RowStatus.FAIL, "non-finite output", job=identity, metrics=metrics)
    return RowResult(job.key, RowStatus.PASS, job=identity, metrics=metrics)


def _parse_args(argv: Sequence[str] | None, config: CompileConfig, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(description))
    add_engine_arguments(parser, config.spec)
    return parser.parse_args(argv)


def compile_modes_check(
    config: CompileConfig,
    script: Path,
    description: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Entry point of a per-model ``--compile`` mode check (driver and child in one)."""
    args = _parse_args(argv, config, description)
    jobs = compile_jobs(args.level)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        job = jobs[args.job_index]
        identity = {"model": config.spec.key, "mode": job.mode.key}
        reason = missing_environment(args.device, args.checkpoint)
        if reason is not None:
            emit_row(RowResult(job.key, RowStatus.SKIP, reason, job=identity))
            return 0
        emit_row(_run_job(config, job, device=args.device, checkpoint=args.checkpoint))
        return 0

    reason = missing_environment(args.device, args.checkpoint)
    if reason is not None:
        return skip(reason, require_gpu=args.require_gpu)

    print(f"\n=== {config.spec.key} compile modes ({args.level.value} level) ===")
    print(f"checkpoint: {args.checkpoint}")
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
    print_table(COLUMNS, rows)
    print()
    print("NOTE: a row that does not run is a regression -- all three backends support --compile,")
    print("      and --compile forces --overlap off.")
    return finish(rows, pass_message=f"{config.spec.key}: every requested compile mode runs")
