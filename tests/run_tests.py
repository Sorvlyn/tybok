"""Run the TyBoK regression checks: every model, at one of two levels.

``quick`` is the everyday gate (minutes): a subset of each check's jobs, fewer replays. ``full`` is
the regression run (tens of minutes, pre-commit / CI): every job, the production replay counts.
Both levels assert the same invariants; see ``tests/README.md``.

Each check runs as its own process, so a check that crashes or runs out of memory cannot take the
run down, and its output streams live. The verdict is ``FAIL`` if any check failed, ``SKIP`` if
every check was skipped (no GPU / no checkpoint), else ``PASS``.

Usage::

    cd TyBoK
    python tests/run_tests.py                                    # all models, quick
    python tests/run_tests.py --models fastwam
    python tests/run_tests.py --models pi05,smolvla --level full
    python tests/run_tests.py --checkpoint fastwam=/path/to/ckpt
    python tests/run_tests.py --dry-run                          # print the commands only

A single check can also be run directly, which takes the same ``--level``::

    python tests/fastwam/graph_replay.py --level full --replays 200
    python tests/smolvla/graph_flags.py --list-jobs
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common import (  # noqa: E402 - after the path bootstrap
    RESULT_PREFIX,
    Column,
    Level,
    RowResult,
    RowStatus,
    check_description,
    finish,
    format_command,
    parse_result_line,
    print_table,
    run_child,
)

__all__ = ["main"]

MODELS = ("fastwam", "pi05", "smolvla")
"""Engine backends: one checkpoint each, so their checks are checkpoint- and GPU-gated."""

GROUPS = ("cli",)
"""Checks that belong to no model (``tests/<group>/``); selected like a model, never gated."""

SELECTABLE = MODELS + GROUPS

SUMMARY_COLUMNS = (
    Column("model", "model", "<"),
    Column("check", "check", "<"),
    Column("level", "level", "<"),
    Column("time (s)", "elapsed_s", "{:.1f}"),
    Column("result", "result", "<"),
)


@dataclass(frozen=True)
class Check:
    """One check script, the model (or model-independent group) it belongs to, and its levels.

    ``levels`` is the runner's share of the level policy: the check itself always runs whatever
    its own ``--level`` selects, but a check whose *cheapest* meaningful run is minutes (compiling
    regions, building an engine per camera count) does not belong in the everyday level. Which
    rows a check runs at a level stays in the check.
    """

    model: str
    name: str
    script: Path
    levels: tuple[Level, ...] = (Level.QUICK, Level.FULL)
    uses_engine: bool = True
    """Whether the check takes the engine options (``--device`` / ``--checkpoint`` / ``--require-gpu``)."""

    @property
    def key(self) -> str:
        return f"{self.model}/{self.name}"


_TESTS = Path(__file__).resolve().parent
_FULL_ONLY = (Level.FULL,)
_BOTH_LEVELS = (Level.QUICK, Level.FULL)


def _check(
    model: str,
    name: str,
    *,
    levels: tuple[Level, ...] = _BOTH_LEVELS,
    uses_engine: bool = True,
) -> Check:
    """The check ``tests/<model>/<name>.py``."""
    return Check(model, name, _TESTS / model / f"{name}.py", levels=levels, uses_engine=uses_engine)


CHECKS: tuple[Check, ...] = (
    # model-independent (no checkpoint, no GPU)
    _check("cli", "surface", uses_engine=False),
    # fastwam
    _check("fastwam", "graph_cameras", levels=_FULL_ONLY),
    _check("fastwam", "graph_replay"),
    _check("fastwam", "overlap_matrix"),
    _check("fastwam", "compile_modes", levels=_FULL_ONLY),
    _check("fastwam", "sweep_geom", uses_engine=False),
    # pi05
    _check("pi05", "graph_cameras"),
    _check("pi05", "overlap_matrix"),
    _check("pi05", "compile_modes", levels=_FULL_ONLY),
    # smolvla
    _check("smolvla", "graph_cameras"),
    _check("smolvla", "overlap_matrix"),
    _check("smolvla", "graph_flags"),
    _check("smolvla", "compile_modes", levels=_FULL_ONLY),
)


@dataclass(frozen=True)
class CheckOutcome:
    """What one check process concluded.

    The verdict comes from the check itself: its exit code plus its own ``RESULT:`` line (a check
    that exits 0 without printing one is treated as a failure). The per-row detail already scrolled
    past above, in the check's own table.
    """

    check: Check
    level: Level
    returncode: int
    result_line: str
    elapsed_s: float

    @property
    def status(self) -> RowStatus:
        if self.returncode != 0:
            return RowStatus.FAIL
        return parse_result_line(self.result_line) or RowStatus.FAIL

    @property
    def detail(self) -> str:
        """The check's own verdict text, for a summary row that is not a plain pass."""
        if self.status is RowStatus.PASS:
            return ""
        if not self.result_line:
            return f"exit {self.returncode} without a RESULT line"
        text = self.result_line.removeprefix(RESULT_PREFIX).strip()
        status_word = self.status.value
        if text.startswith(status_word):
            text = text[len(status_word) :].strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return text.strip() or status_word

    def summary_row(self) -> RowResult:
        """One row for the run's summary table."""
        metrics = {
            "model": self.check.model,
            "check": self.check.name,
            "level": self.level.value,
            "elapsed_s": self.elapsed_s,
            "result": self.status.value,
        }
        return RowResult(self.check.key, self.status, self.detail, metrics=metrics)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(__doc__))
    parser.add_argument(
        "--models",
        default="all",
        help=f"comma-separated subset of {', '.join(SELECTABLE)}, or 'all' (default: all)",
    )
    parser.add_argument(
        "--level",
        type=Level,
        choices=("quick", "full"),
        metavar="{quick,full}",
        default=Level.QUICK,
        help="quick = the everyday subset; full = the regression run (default: quick)",
    )

    parser.add_argument("--device", default="cuda", help="torch device for every check")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="override one model's checkpoint (repeatable)",
    )

    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="FAIL instead of SKIP when CUDA or a checkpoint is unavailable",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would run, then exit",
    )

    return parser.parse_args(argv)


class UsageError(Exception):
    """A command line this runner cannot act on (exit code 2, as ``argparse`` does)."""


def _resolve_models(requested: str) -> list[str]:
    if requested == "all":
        return list(SELECTABLE)
    models = [model.strip() for model in requested.split(",") if model.strip()]
    unknown = [model for model in models if model not in SELECTABLE]
    if unknown:
        raise UsageError(f"unknown model: {', '.join(unknown)} (expected {', '.join(SELECTABLE)} or all)")

    return models


def _resolve_checkpoints(overrides: Sequence[str]) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    for override in overrides:
        model, separator, path = override.partition("=")
        if not separator or model not in MODELS or not path:
            raise UsageError(f"expected --checkpoint MODEL=PATH, got {override!r}")
        checkpoints[model] = path
    return checkpoints


def _command(check: Check, args: argparse.Namespace, checkpoints: dict[str, str]) -> list[str]:
    command = ["--level", args.level.value]
    if not check.uses_engine:
        return command
    command += ["--device", args.device]
    if check.model in checkpoints:
        command += ["--checkpoint", checkpoints[check.model]]
    if args.require_gpu:
        command += ["--require-gpu"]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Run every selected check and report one summary table."""
    args = _parse_args(argv)
    try:
        models = _resolve_models(args.models)
        checkpoints = _resolve_checkpoints(args.checkpoint)
    except UsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    checks = [check for check in CHECKS if check.model in models and args.level in check.levels]
    skipped = [check for check in CHECKS if check.model in models and args.level not in check.levels]

    if args.dry_run:
        print(f"level: {args.level.value}; models: {', '.join(models)}")
        for check in checks:
            print(format_command(check.script, _command(check, args, checkpoints)))
        for check in skipped:
            print(f"# not run at {args.level.value} level: {check.key}")
        return 0

    outcomes: list[CheckOutcome] = []
    for check in checks:
        command = _command(check, args, checkpoints)
        print(f"\n{'=' * 78}")
        print(f"=== {check.key} ({args.level.value})")
        print(f"=== {format_command(check.script, command)}")
        print("=" * 78, flush=True)
        started = time.monotonic()
        child = run_child(check.script, command)
        elapsed = time.monotonic() - started
        outcomes.append(CheckOutcome(check, args.level, child.returncode, child.result_line, elapsed))

    print()
    print_table(SUMMARY_COLUMNS, [outcome.summary_row() for outcome in outcomes])
    if skipped:
        print()
        print(f"not run at {args.level.value} level (use --level full): " + ", ".join(check.key for check in skipped))
    failed = [outcome for outcome in outcomes if outcome.status is RowStatus.FAIL]
    if failed:
        print()
        print("failed checks:")
        for outcome in failed:
            print(f"  {outcome.check.key}: {outcome.detail}")
    return finish(
        [outcome.summary_row() for outcome in outcomes],
        pass_message=f"all {len(outcomes)} checks",
    )


if __name__ == "__main__":
    sys.exit(main())
