"""smolvla: does every flag that reaches the graph path stay bit-exact vs eager?

``--graph`` for smolvla captures ``VLAFlowMatching.sample_actions`` -- the whole inference core,
with the ``prefill_layer`` x ``step0_layer`` fusion inside it when ``overlap`` is set -- as **one
graph per shape**, keyed only by ``(n_cams,)``. Several flags change what that core *does* rather
than just its shapes, so they are invisible to the graph key and to the overlap matrix (which runs
the default flag set only):

``--no-expert-prefix-kv-cache``   projections inline instead of cached (branch inside the core)
``--sampler heun``                two velocity evaluations per step (forces overlap off)
``--steps N``                     the loop length is baked into the captured core
``--seed N``                      implicit noise is drawn inside the core; the RNG order must match
``select_action``                 the graph-mode action queue (refill + pops)

This gate crosses the flag set with graph on/off; a row passes when the two processes are
**bit-identical**. Every engine runs in its own process: a captured graph owns a private CUDA pool.

Why only smolvla has this check
-------------------------------
It is the only backend whose knobs both live *inside* the captured core and stay out of the graph
key. pi05 and fastwam only support ``--sampler euler`` (``pi05/engine.py``, ``fastwam/engine.py``
raise on anything else), fastwam puts ``steps`` into the graph key, and fastwam rejects
``--no-expert-prefix-kv-cache`` outright. Their fused-tier flags are already crossed with graph
mode by the overlap matrices. See ``tests/README.md``.

Rows, and the jobs of a level (``--list-jobs``): each row runs twice -- ``graph=False`` for the
eager reference, ``graph=True`` -- for ``predict_action_chunk``, plus the action queue
(``select_action``, 5 requests) on the baseline row. ``quick`` runs ``baseline`` and
``no-kv-cache``; ``full`` runs every row.

Implicit noise (no explicit tensor) is only comparable across processes when ``--seed`` pins the
engine RNG: the graph path prefetches the next request's noise, so without a seed the two
processes sit at different points in the global RNG stream. Every other row passes explicit noise,
so the comparison isolates the flag under test.

Usage::

    cd TyBoK
    python tests/smolvla/graph_flags.py
    python tests/smolvla/graph_flags.py --level full
    python tests/smolvla/graph_flags.py --list-jobs
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Allow ``python tests/smolvla/graph_flags.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common import (  # noqa: E402 - after the path bootstrap
    Column,
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
from tests.smolvla.spec import SMOLVLA  # noqa: E402 - after the path bootstrap

__all__ = ["main"]

#: ``select_action`` requests per job (one chunk refill plus pops).
SELECT_REQUESTS = 5

COLUMNS = (
    Column("row", "key", "<"),
    Column("op", "op", "<"),
    Column("overlap", "overlap"),
    Column("kv-cache", "kv_cache"),
    Column("sampler", "sampler"),
    Column("steps", "steps"),
    Column("noise", "noise"),
    Column("max|graph-eager|", "max_abs_diff_vs_eager", "{:.3e}"),
    Column("result", "status", "<"),
)


class Op(str, Enum):
    """Which request path to exercise."""

    PREDICT = "predict"
    SELECT = "select"


@dataclass(frozen=True)
class FlagRow:
    """One flag combination under test."""

    name: str
    engine_kwargs: Mapping[str, Any]
    implicit_noise: bool = False
    quick: bool = False


@dataclass(frozen=True)
class FlagJob:
    """One process: a flag row with graph off (the reference) or graph on."""

    row: FlagRow
    graph: bool
    op: Op
    quick: bool

    @property
    def key(self) -> str:
        return f"{self.row.name} [{self.op.value}, {'graph' if self.graph else 'eager'}]"


ROWS: tuple[FlagRow, ...] = (
    FlagRow("baseline", {}, quick=True),
    FlagRow("no-kv-cache", {"cache_expert_prefix_kv": False}, quick=True),
    FlagRow("no-kv-cache+no-overlap", {"cache_expert_prefix_kv": False, "overlap": False}),
    FlagRow("heun", {"sampler": "heun"}),
    FlagRow("heun+no-kv-cache", {"sampler": "heun", "cache_expert_prefix_kv": False}),
    FlagRow("heun+no-overlap", {"sampler": "heun", "overlap": False}),
    FlagRow("steps-4", {"num_steps": 4}),
    FlagRow("heun+steps-4", {"sampler": "heun", "num_steps": 4}),
    FlagRow("implicit-noise+seed-7", {"seed": 7}, implicit_noise=True),
)

JOBS: tuple[FlagJob, ...] = tuple(
    FlagJob(row, graph, Op.PREDICT, quick=row.quick) for row in ROWS for graph in (False, True)
) + tuple(
    # the action queue: full level only (it adds an engine load and 5 requests per process)
    FlagJob(ROWS[0], graph, Op.SELECT, quick=False)
    for graph in (False, True)
)


def _pairs(jobs: Sequence[FlagJob]) -> list[tuple[FlagRow, Op]]:
    """The distinct ``(row, op)`` pairs of ``jobs``, in declaration order."""
    pairs: list[tuple[FlagRow, Op]] = []
    for job in jobs:
        if (job.row, job.op) not in pairs:
            pairs.append((job.row, job.op))
    return pairs


def _noise(spec: Mapping[str, Any], seed: int) -> Any:
    import torch

    return torch.randn(1, spec["chunk_size"], spec["action_dim"], generator=torch.Generator().manual_seed(seed))


def _run_job(job: FlagJob, *, checkpoint: str, device: str) -> RowResult:
    """Child side: build the engine for one row, run one request path, send back the output."""
    import numpy as np

    from tybok.registry import create_engine

    identity = {"row": job.row.name, "op": job.op.value, "graph": int(job.graph)}
    metrics: dict[str, Any] = {
        "op": job.op.value,
        "overlap": None,
        "kv_cache": None,
        "sampler": None,
        "steps": None,
        "noise": "implicit" if job.row.implicit_noise else "explicit",
    }

    try:
        engine = create_engine(
            checkpoint,
            model_type="smolvla",
            device=device,
            graph=job.graph,
            warmup=True,
            **job.row.engine_kwargs,
        )

        # ``PolicyEngine`` declares neither ``policy`` nor ``config`` (every backend sets them),
        # hence getattr throughout.
        model = getattr(getattr(engine, "policy", None), "model", None)
        metrics["overlap"] = bool(getattr(engine, "overlap", False))
        metrics["kv_cache"] = bool(getattr(model, "cache_expert_prefix_kv", None))
        metrics["sampler"] = getattr(model, "sampler", "-")
        metrics["steps"] = int(getattr(getattr(engine, "config", None), "num_steps", 0))

        spec = engine.describe()
        frame = engine._profile_frame(seed=1)
        if job.op is Op.PREDICT:
            noise = None if job.row.implicit_noise else _noise(spec, 123)
            output = np.asarray(engine.predict_action_chunk(frame, noise=noise), dtype=np.float64)
        else:
            chunks = [
                np.asarray(
                    engine.select_action(frame, noise=None if job.row.implicit_noise else _noise(spec, 1000 + index)),
                    dtype=np.float64,
                )
                for index in range(SELECT_REQUESTS)
            ]
            output = np.stack(chunks)
    except Exception as error:  # noqa: BLE001 - a row that cannot run is a result, not a crash
        return RowResult(
            job.key,
            RowStatus.FAIL,
            f"error: {type(error).__name__}: {error}",
            job=identity,
            metrics=metrics,
        )

    return RowResult(
        job.key,
        RowStatus.PASS,
        job=identity,
        metrics=metrics,
        values=[float(value) for value in output.ravel()],
    )


def _find(rows: Sequence[RowResult], *, row: FlagRow, op: Op, graph: bool) -> RowResult | None:
    wanted = {"row": row.name, "op": op.value, "graph": int(graph)}
    for result in rows:
        if all(result.job.get(field) == value for field, value in wanted.items()):
            return result
    return None


def _judge(row: FlagRow, op: Op, rows: Sequence[RowResult]) -> RowResult:
    """Compare a row's graph process against its eager reference."""
    import numpy as np

    eager = _find(rows, row=row, op=op, graph=False)
    graph = _find(rows, row=row, op=op, graph=True)
    display = graph if graph is not None and graph.is_pass else eager
    metrics: dict[str, Any] = dict(display.metrics) if display is not None else {}
    metrics["op"] = op.value
    key = f"{row.name} ({op.value})" if op is Op.SELECT else row.name

    for name, result in (("eager", eager), ("graph", graph)):
        if result is None or not result.is_pass:
            detail = result.detail if result is not None else "row missing"
            return RowResult(key, RowStatus.FAIL, f"{name} failed ({detail})", metrics=metrics)
    if eager is None or graph is None:
        return RowResult(key, RowStatus.FAIL, "row missing", metrics=metrics)
    if eager.values is None or graph.values is None:
        return RowResult(key, RowStatus.FAIL, "missing output", metrics=metrics)

    difference = float(
        np.abs(np.asarray(graph.values, dtype=np.float64) - np.asarray(eager.values, dtype=np.float64)).max()
    )

    metrics["max_abs_diff_vs_eager"] = difference
    status = RowStatus.PASS if difference == 0.0 else RowStatus.FAIL
    detail = "" if difference == 0.0 else f"differs by {difference:.3e}"
    return RowResult(key, status, detail, metrics=metrics)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(__doc__))
    add_engine_arguments(parser, SMOLVLA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the flag matrix (driver), or one job of it (``--job-index``)."""
    args = _parse_args(argv)
    jobs = jobs_for_level(JOBS, args.level)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        job = jobs[args.job_index]
        identity = {"row": job.row.name, "op": job.op.value, "graph": int(job.graph)}
        reason = missing_environment(args.device, args.checkpoint)
        if reason is not None:
            emit_row(RowResult(job.key, RowStatus.SKIP, reason, job=identity))
            return 0
        emit_row(_run_job(job, checkpoint=args.checkpoint, device=args.device))
        return 0

    reason = missing_environment(args.device, args.checkpoint)
    if reason is not None:
        return skip(reason, require_gpu=args.require_gpu)

    print(f"\n=== smolvla graph flags ({args.level.value} level) ===")
    print(f"checkpoint: {args.checkpoint}")
    rows = run_jobs(
        Path(__file__),
        job_keys(jobs),
        ["--level", args.level.value, "--device", args.device, "--checkpoint", args.checkpoint],
    )

    judged = [_judge(row, op, rows) for row, op in _pairs(jobs)]
    print()
    print_table(COLUMNS, judged)
    print()
    print("NOTE: a row passes when graph and eager are bit-identical -- a flag silently ignored")
    print("      under --graph shows up here as a non-zero deviation.")
    return finish(judged, pass_message="smolvla: every flag bit-exact under --graph")


if __name__ == "__main__":
    sys.exit(main())
