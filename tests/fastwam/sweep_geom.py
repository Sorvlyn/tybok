"""fastwam kernel-sweep self-test: the pure logic of ``kernels/sweep.py`` must be right.

No GPU, no compilation, no checkpoint -- this is a plain logic test, and it exists because the
expensive failure mode of a parameter sweep is not a crash but **silently bogus numbers**: a macro
that was never patched (every candidate compiles to the same geometry, so "all configs identical,
same speed"), an axis name bound to the wrong value slot, or a structural change swept as if it
were a parameter. Four groups:

  1. **Macro patching is reversible and in place** -- feeding the current values back through
     ``patch_macro`` must reproduce the source file byte for byte (same values => same text),
     patching one value must parse back, patching it back must equal the original, and a value
     count that does not match the template must be rejected. (An earlier version regenerated the
     ``#define`` instead; 11 macros wrap their values differently -- first lines 108-121 chars, no
     uniform width rule -- so regeneration never matched.)
  2. **Axis names line up with value slots** -- the axis names parsed from the template signature
     must line up with each macro's i-th value (checked against known geometries), otherwise
     ``--axes "BN=64"`` patches some other slot.
  3. **Structural axes are refused** -- PERSIST/SPLIT/EPI/... cannot be swept by patching the macro
     (PERSIST was measured writing the whole output as NaN), so ``--axes`` must reject them with a
     readable reason, while block axes like BN stay sweepable.
  4. **Candidate generation and the smem warning** -- cartesian product, baseline always first,
     duplicate values de-duplicated; the smem estimate must match the byte counts written in the
     source comments (F_PHASE_4=48384, F_PHASE_2=49152).

All four groups run at both levels; the whole check is a couple of seconds of CPU work.

Usage::

    cd tybok
    python tests/fastwam/sweep_geom.py
    python tests/fastwam/sweep_geom.py --list-jobs
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Allow ``python tests/fastwam/sweep_geom.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common import (  # noqa: E402 - after the path bootstrap
    Column,
    RowResult,
    RowStatus,
    add_level_arguments,
    check_description,
    emit_row,
    finish,
    job_keys,
    jobs_for_level,
    print_job_list,
    print_table,
    run_jobs,
)

__all__ = ["main"]

COLUMNS = (
    Column("group", "key", "<"),
    Column("verified", "verified", "<"),
    Column("result", "status", "<"),
)


class Group(str, Enum):
    """The four assertion groups of ``kernels/sweep.py``."""

    PATCH = "macro patching"
    AXES = "axis names"
    STRUCTURAL = "structural axes"
    CANDIDATES = "candidates + smem"


@dataclass(frozen=True)
class SweepJob:
    """One group, run in its own process like every other job in this suite."""

    group: Group

    @property
    def quick(self) -> bool:
        return True

    @property
    def key(self) -> str:
        return self.group.value


JOBS: tuple[SweepJob, ...] = tuple(SweepJob(group) for group in Group)


def _check(failures: list[str], condition: bool, message: str) -> None:
    """Record ``message`` when ``condition`` is false."""
    if not condition:
        failures.append(message)


def _run_patch(sweep: Any) -> tuple[str, list[str]]:
    """Group 1: ``patch_macro`` is in place, reversible and validates the value count."""
    failures: list[str] = []
    for name, slot in sorted(sweep.slots().items()):
        original = (sweep._KERNEL_DIR / slot.file).read_text()
        values = list(sweep.current_values(slot))
        if len(values) != len(slot.axes):
            failures.append(f"{name}: parsed {len(values)} values for {len(slot.axes)} template axes")

            continue
        _check(
            failures,
            sweep.patch_macro(original, slot.macro, values) == original,
            f"{name}: patching the current values changed the text (not in place)",
        )

        tweaked = list(values)
        tweaked[-1] = 1 - tweaked[-1]  # only the last axis (a 0/1 switch)
        patched = sweep.patch_macro(original, slot.macro, tweaked)
        match = sweep._define_re(slot.macro).search(patched)
        body, _ = sweep._read_define_body(patched, match)
        _check(
            failures,
            [int(value) for value in re.findall(r"-?\d+", body)] == tweaked,
            f"{name}: patched values do not parse back",
        )

        _check(
            failures,
            sweep.patch_macro(patched, slot.macro, values) == original,
            f"{name}: patching back does not reproduce the original",
        )

        try:  # a value count that does not match must be rejected
            sweep.patch_macro(original, slot.macro, values[:-1])
        except SystemExit:
            pass
        else:
            failures.append(f"{name}: a missing value was accepted instead of rejected")
    return f"{len(sweep.slots())} macros: in place, reversible, count checked", failures


def _run_axes(sweep: Any) -> tuple[str, list[str]]:
    """Group 2: axis names and value slots line up."""
    failures: list[str] = []
    slots = sweep.slots()
    expected = {
        "vdit.ffn/F_PHASE_2": {"BM": 64, "BN": 64, "XS": 1, "PERSIST": 1, "EPI": 1},
        "vdit.ffn/F_PHASE_4": {"BM": 64, "BN": 48, "XS": 2, "PERSIST": 1, "RESID": 1},
        "tmt5.ffn/F_PHASE_2": {"BM": 128, "BN": 64, "BK": 128},
        "tmt5.attn/S_PHASE_5": {"BM": 128, "BN": 64, "BK": 128},
    }
    for name, geometry in expected.items():
        slot = slots[name]
        values = dict(zip(slot.axes, sweep.current_values(slot)))
        for axis, want in geometry.items():
            _check(
                failures,
                values.get(axis) == want,
                f"{name}: {axis} reads as {values.get(axis)}, expected {want} (axis order?)",
            )

    _check(
        failures,
        len(slots["vdit.ffn/F_PHASE_4"].axes) == 29,
        "the F_PHASE_4 template does not have 29 axes",
    )

    return f"{len(slots)} slots, 4 geometry probes matched", failures


def _run_structural(sweep: Any) -> tuple[str, list[str]]:
    """Group 3: structural axes are refused, block axes stay sweepable."""
    failures: list[str] = []
    slot = sweep.slots()["vdit.ffn/F_PHASE_4"]
    baseline = sweep.current_values(slot)
    refused = 0
    for axis in sweep._STRUCTURAL:
        if axis not in slot.axes:
            continue
        index = slot.axes.index(axis)
        changed = tuple(list(baseline[:index]) + [1 - baseline[index]] + list(baseline[index + 1 :]))
        if changed == baseline:
            continue
        refused += 1
        try:
            sweep._check_sweepable(slot, baseline, changed, False)
        except SystemExit as error:
            _check(failures, axis in str(error), f"{axis}: refused without naming the axis")
        else:
            failures.append(f"{axis}: not refused")
        sweep._check_sweepable(slot, baseline, changed, True)  # --allow-structural lets it through
    index = slot.axes.index("BN")
    sweep._check_sweepable(slot, baseline, tuple(list(baseline[:index]) + [32] + list(baseline[index + 1 :])), False)
    return f"{refused} structural axes refused, BN allowed", failures


def _run_candidates(sweep: Any) -> tuple[str, list[str]]:
    """Group 4: candidate generation and the smem estimate."""
    failures: list[str] = []
    slots = sweep.slots()
    slot = slots["vdit.ffn/F_PHASE_4"]
    baseline = sweep.current_values(slot)
    candidates = sweep._candidates([], "BN=32,48,64 BK=64,128", slot, baseline)
    _check(failures, candidates[0] == tuple(baseline), "the baseline is not the first candidate")
    _check(
        failures,
        len(candidates) == 6,
        f"expected 6 de-duplicated candidates (3x2, one of them the baseline), got {len(candidates)}",
    )

    bn_index, bk_index = slot.axes.index("BN"), slot.axes.index("BK")
    _check(failures, {c[bn_index] for c in candidates} == {32, 48, 64}, "BN values are incomplete")
    _check(failures, {c[bk_index] for c in candidates} == {64, 128}, "BK values are incomplete")
    _check(
        failures,
        sweep._candidates([",".join(str(value) for value in baseline)], "", slot, baseline) == [tuple(baseline)],
        "--tiles given the current values does not de-duplicate against the baseline",
    )

    duplicate = list(baseline)
    duplicate[bn_index] = 48
    _check(
        failures,
        len(sweep._candidates([",".join(map(str, duplicate))], "", slot, baseline)) == 1,
        "--tiles repeating the current geometry does not de-duplicate",
    )

    up = dict(zip(slots["vdit.ffn/F_PHASE_2"].axes, sweep.current_values(slots["vdit.ffn/F_PHASE_2"])))
    up_estimate = sweep._smem_estimate(slots["vdit.ffn/F_PHASE_2"], tuple(up.values()))
    down_estimate = sweep._smem_estimate(slot, baseline)
    _check(failures, up_estimate == 49152, f"F_PHASE_2 smem estimates {up_estimate}, comment says 49152")
    _check(failures, down_estimate == 48384, f"F_PHASE_4 smem estimates {down_estimate}, comment says 48384")
    budget = sweep._smem_budget(slot.kernel)
    _check(failures, budget == 49152, f"F_PHASE_4 budget reads {budget}")
    return "6 candidates, baseline first, de-duplicated, smem matches the comments", failures


_GROUPS = {
    Group.PATCH: _run_patch,
    Group.AXES: _run_axes,
    Group.STRUCTURAL: _run_structural,
    Group.CANDIDATES: _run_candidates,
}


def _run_group(job: SweepJob) -> RowResult:
    """Child side: run one assertion group."""
    from tybok.policies.fastwam.kernels import sweep

    verified, failures = _GROUPS[job.group](sweep)
    metrics: dict[str, Any] = {"verified": verified, "checks": len(failures)}
    if failures:
        return RowResult(
            job.key,
            RowStatus.FAIL,
            "; ".join(failures),
            job={"group": job.group.value},
            metrics=metrics,
        )

    return RowResult(job.key, RowStatus.PASS, job={"group": job.group.value}, metrics=metrics)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=check_description(__doc__))
    add_level_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep self-test (driver), or one group of it (``--job-index``)."""
    args = _parse_args(argv)
    jobs = jobs_for_level(JOBS, args.level)

    if args.list_jobs:
        print_job_list(job_keys(jobs))
        return 0
    if args.job_index is not None:
        emit_row(_run_group(jobs[args.job_index]))
        return 0

    print(f"\n=== fastwam kernel-sweep self-test ({args.level.value} level) ===")
    print("no GPU, no compilation, no checkpoint: pure logic of kernels/sweep.py")
    rows = run_jobs(Path(__file__), job_keys(jobs), ["--level", args.level.value])
    print()
    print_table(COLUMNS, rows)
    print()
    print("NOTE: a green sweep can still be worthless if a macro was never patched -- these four")
    print("      groups pin the ways that happens silently.")
    return finish(rows, pass_message="fastwam: sweep logic pinned in all four groups")


if __name__ == "__main__":
    sys.exit(main())
