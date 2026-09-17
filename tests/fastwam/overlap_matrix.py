"""fastwam overlap matrix: kernel tier x graph mode.

Rows (every one requested with ``overlap=True``):

  1. ``eager | torch``
  2. ``graph | torch``
  3. ``eager | fused split``
  4. ``eager | fused coop``
  5. ``graph | fused split``
  6. ``graph | fused coop``

For each row the engine is built in exactly that configuration and the check verifies

  * the overlap branch is **actually taken** -- the old failure mode was overlap being silently
    forced off in graph mode;
  * ``overlap`` and ``no overlap`` agree **bit-exactly within the same kernel tier** (overlap only
    reorders the step-0 action layers against the video prefill, so it must not change a single
    bit; the fused tiers are *not* bit-exact against the torch tier, which is why each row is its
    own control);
  * for the graph rows, the captured graph agrees bit-exactly with the eager path;
  * the fused form that really ran (cooperative vs split) is reported as measured.

Each row is its own process: the tiers keep large resident workspaces and the caching allocator
does not return them reliably across in-process engine rebuilds (row 4 OOMs when everything
shares one process).

``quick`` runs the production tier (``graph | fused coop``); ``full`` adds the other five rows
(the torch tier, the non-cooperative fused form, and the eager rows that exercise overlap without
a graph).

Usage::

    cd tybok
    python tests/fastwam/overlap_matrix.py
    python tests/fastwam/overlap_matrix.py --level full --checkpoint /path/to/ckpt
    python tests/fastwam/overlap_matrix.py --list-jobs
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow ``python tests/fastwam/overlap_matrix.py`` without ``pip install -e .``.
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
from tests.fastwam.spec import (  # noqa: E402 - after the path bootstrap
    FASTWAM,
    FUSED_COOP_TIER,
    FUSED_SPLIT_TIER,
    TORCH_TIER,
)

__all__ = ["main"]

COLUMNS = (
    Column("row", "key", "<"),
    Column("overlap", "overlap_engaged"),
    Column("form", "kernel_form"),
    Column("graphs", "graphs"),
    Column("d(config, overlap)", "max_abs_diff_vs_config", "{:.3e}"),
    Column("d(overlap, seq)", "max_abs_diff_vs_sequential", "{:.3e}"),
    Column("result", "status", "<"),
)


@dataclass(frozen=True)
class MatrixJob:
    """One (kernel tier, graph mode) configuration."""

    label: str
    graph: bool
    tier: dict[str, Any]
    quick: bool

    @property
    def key(self) -> str:
        return self.label


JOBS: tuple[MatrixJob, ...] = (
    MatrixJob("eager | torch", graph=False, tier=TORCH_TIER, quick=False),
    MatrixJob("graph | torch", graph=True, tier=TORCH_TIER, quick=False),
    MatrixJob("eager | fused split", graph=False, tier=FUSED_SPLIT_TIER, quick=False),
    MatrixJob("eager | fused coop", graph=False, tier=FUSED_COOP_TIER, quick=False),
    MatrixJob("graph | fused split", graph=True, tier=FUSED_SPLIT_TIER, quick=False),
    # the production tier and the only graph row the quick level needs
    MatrixJob("graph | fused coop", graph=True, tier=FUSED_COOP_TIER, quick=True),
)


def _relative_difference(actual: Any, expected: Any) -> float:
    import numpy as np

    return float(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)).max())


def _measured_kernel_form(engine: Any, job: MatrixJob) -> str:
    """What actually ran: ``torch``, ``split`` or ``coop``."""
    tier = job.tier
    if not tier.get("video_fused") and not tier.get("action_fused"):
        return "torch"
    splits = [
        bool(getattr(runner, "_split", True))
        for name in ("_video_fused_runner", "_video_fused_attn_runner", "_action_fused_runner")
        if (runner := getattr(engine, name, None)) is not None
    ]
    if not splits:
        return "torch"
    return "split" if any(splits) else "coop"


def _run_row(job: MatrixJob, *, checkpoint: str, device: str, text_encoder_device: str) -> RowResult:
    """Child side: build the tier, compare the configured path against its in-tier references."""
    import torch

    from tybok.registry import create_engine

    engine = create_engine(
        checkpoint,
        model_type="fastwam",
        device=device,
        warmup=True,
        text_encoder_device=text_encoder_device,
        graph=job.graph,
        overlap=True,
        **job.tier,
    )

    # ``PolicyEngine`` does not declare ``policy`` (every backend sets it), hence getattr.
    model = getattr(getattr(engine, "policy", None), "model", None)
    if model is None:  # cannot happen for fastwam; a missing model is a result, not a crash
        return RowResult(job.key, RowStatus.FAIL, "engine has no policy model")

    # Deterministic inputs, built once per row: the VAE / cuDNN algorithm choice is stable within
    # a process, so every comparison below is exact.
    spec = engine.describe()
    generator = torch.Generator().manual_seed(1)
    images = {
        camera.split("observation.images.")[-1]: torch.rand(
            3, spec["resize"][1], spec["resize"][0], generator=generator
        )
        for camera in spec["cameras"]
    }
    frame = engine.make_frame(
        images, torch.rand(engine.config.proprio_dim or 8, generator=generator), "pick up the cup"
    )

    noise = torch.randn(1, spec["chunk_size"], spec["action_dim"], generator=torch.Generator().manual_seed(123))

    # (1) the configured path: graph replay when graph=True, else eager overlap.
    output_configured = engine.predict_action_chunk(frame, noise)
    metrics: dict[str, Any] = {
        "overlap_engaged": int(model._overlap_stream is not None),
        "kernel_form": _measured_kernel_form(engine, job),
        "graphs": int(engine._graph_runner.num_graphs) if engine._graph_runner is not None else 0,
    }

    # (2) eager references in the *same* kernel tier: detach the graph, toggle the branch.
    model._graph_runner = None
    model.overlap = True
    output_overlap = engine.predict_action_chunk(frame, noise)
    model.overlap = False
    output_sequential = engine.predict_action_chunk(frame, noise)

    metrics["max_abs_diff_vs_config"] = _relative_difference(output_configured, output_overlap)
    metrics["max_abs_diff_vs_sequential"] = _relative_difference(output_overlap, output_sequential)

    failed = metrics["overlap_engaged"] == 0 or any(
        metrics[name] != 0.0 for name in ("max_abs_diff_vs_config", "max_abs_diff_vs_sequential")
    )

    detail = "overlap not engaged" if metrics["overlap_engaged"] == 0 else "not bit-exact vs its tier"

    return RowResult(
        job.key,
        RowStatus.FAIL if failed else RowStatus.PASS,
        detail if failed else "",
        job={"row": job.label, "graph": int(job.graph)},
        metrics=metrics,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(__doc__))
    add_engine_arguments(parser, FASTWAM)
    parser.add_argument("--text-encoder-device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix (driver), or one row of it (``--job-index``)."""
    args = _parse_args(argv)
    jobs = jobs_for_level(JOBS, args.level)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        job = jobs[args.job_index]
        reason = missing_environment(args.device, args.checkpoint)
        if reason is not None:
            emit_row(
                RowResult(
                    job.key,
                    RowStatus.SKIP,
                    reason,
                    job={"row": job.label, "graph": int(job.graph)},
                )
            )
            return 0
        emit_row(
            _run_row(
                job,
                checkpoint=args.checkpoint,
                device=args.device,
                text_encoder_device=args.text_encoder_device,
            )
        )
        return 0

    reason = missing_environment(args.device, args.checkpoint)
    if reason is not None:
        return skip(reason, require_gpu=args.require_gpu)

    print(f"\n=== fastwam overlap matrix ({args.level.value} level) ===")
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
        ],
    )

    print()
    print_table(COLUMNS, rows)
    print()
    print("NOTE: the fused tiers are a quantisation-drift tier, so every comparison stays inside")
    print("      one tier; overlap must not change a bit within a tier.")
    return finish(rows, pass_message="fastwam: overlap supported in every row")


if __name__ == "__main__":
    sys.exit(main())
