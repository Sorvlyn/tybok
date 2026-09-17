"""Running one job per process.

Every check that builds engines runs **one job per process**: tiered kernels keep large resident
workspaces, a captured CUDA graph owns a private memory pool, and the caching allocator does not
hand those back reliably across in-process engine rebuilds (rows of the fastwam matrix OOM when
they share a process). So a check is a driver that spawns itself with ``--job-index N`` and reads
back the ``ROW`` line the child emits (:mod:`tests._common.rows`).

Child output is streamed while it runs -- a ``full`` run takes tens of minutes and must show
progress -- and the ``ROW`` lines are consumed rather than echoed, because the driver prints the
same information as a table at the end.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .rows import RESULT_PREFIX, RowResult, RowStatus, parse_result_line, parse_row_line

__all__ = ["PYTHON", "REPO_ROOT", "ChildResult", "format_command", "run_child", "run_jobs"]

#: Repository root (``TyBoK/``), the working directory every job runs in.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Interpreter used for the job processes -- the one running the driver, not ``python``.
PYTHON = sys.executable or "python"


@dataclass(frozen=True)
class ChildResult:
    """What a single job process produced."""

    returncode: int
    rows: list[RowResult] = field(default_factory=list)
    result_line: str = ""
    tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def status(self) -> RowStatus:
        """Verdict of the child as a whole, from its exit code and ``RESULT:`` line.

        A child that exits non-zero, or that prints no ``RESULT:`` line at all, is a failure: a
        check must say what it concluded.
        """
        if self.returncode != 0:
            return RowStatus.FAIL
        return parse_result_line(self.result_line) or RowStatus.FAIL


def format_command(script: Path | str, args: Sequence[str]) -> str:
    """The command line of a job, for ``--dry-run`` and error messages."""
    return " ".join([PYTHON, str(script), *args])


def run_child(script: Path | str, args: Sequence[str], *, stream: bool = True) -> ChildResult:
    """Run one job process, echoing its output as it arrives.

    ``stderr`` is merged into ``stdout`` so the two stay ordered and a crash is visible in the
    captured tail. Rows are collected; the last ``RESULT:`` line and the last output line are
    kept for the driver's error reporting.
    """
    command = [PYTHON, str(script), *args]
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    rows: list[RowResult] = []
    result_line = ""
    tail = ""
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line.strip():
            tail = line
        row = parse_row_line(line)
        if row is not None:
            rows.append(row)
            continue
        if line.startswith(RESULT_PREFIX):
            result_line = line
        if stream:
            print(line, flush=True)

    return ChildResult(returncode=process.wait(), rows=rows, result_line=result_line, tail=tail)


def run_jobs(
    script: Path | str,
    job_keys: Sequence[str],
    args: Sequence[str],
    *,
    stream: bool = True,
) -> list[RowResult]:
    """Run every job of ``script`` as its own process and collect one row per job.

    ``job_keys`` are the display keys of the jobs in the order the child enumerates them (the
    child rebuilds the same list from the same ``--level``). A job that produces no row -- a
    crash, an OOM, a killed process -- becomes a ``FAIL`` row carrying the child's last output
    line, so a broken job can never be mistaken for a passing one.
    """
    rows: list[RowResult] = []
    for index, key in enumerate(job_keys):
        print(f"[{index + 1}/{len(job_keys)}] {key} ...", flush=True)
        child = run_child(script, [*args, "--job-index", str(index)], stream=stream)
        if child.rows:
            rows.extend(child.rows)
            continue
        rows.append(
            RowResult(
                key=key,
                status=RowStatus.FAIL,
                detail=f"no result row (exit {child.returncode}): {child.tail}".strip(),
            )
        )

    return rows
