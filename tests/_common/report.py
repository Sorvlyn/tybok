"""Uniform reporting: declarative tables, one ``RESULT:`` line, one exit code.

Checks report rows (:class:`tests._common.rows.RowResult`) and describe their table as a list of
:class:`Column`, so every check prints the same shape of table and the same verdict line without
repeating formatting code.

Verdict policy (identical in every check, and in :mod:`tests.run_tests`):

* any ``FAIL`` row       -> ``RESULT: FAIL``, exit 1
* every row ``SKIP``     -> ``RESULT: SKIP (...)``, exit 0 (no GPU / no checkpoint)
* otherwise              -> ``RESULT: PASS (...)``, exit 0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .rows import RowResult

__all__ = ["Column", "finish", "print_table", "skip"]

#: Longest status cell (a failing row's detail is truncated to keep the table readable).
_MAX_STATUS = 72


@dataclass(frozen=True)
class Column:
    """One table column: its header and where the cell text comes from."""

    header: str
    key: str
    """``key`` / ``status`` / ``detail`` of the row, or a ``metrics`` entry."""

    fmt: str = "{}"
    """``str.format`` template, e.g. ``"{:.3e}"`` for a deviation."""

    align: str = ">"
    """``<`` / ``>`` / ``^``."""


def _cell(row: RowResult, column: Column) -> str:
    if column.key == "key":
        value: Any = row.key
    elif column.key == "status":
        value = row.status.value if row.is_pass else f"{row.status.value} {row.detail}".strip()
        if len(value) > _MAX_STATUS:
            value = value[: _MAX_STATUS - 1].rstrip() + "\u2026"
    elif column.key == "detail":
        value = row.detail
    else:
        value = row.metrics.get(column.key)
        if value is None:
            return "-"
    if isinstance(value, float):
        return column.fmt.format(value)
    return str(value)


def print_table(columns: Sequence[Column], rows: Sequence[RowResult]) -> None:
    """Print ``rows`` under ``columns``, sized to the content."""
    headers = [column.header for column in columns]
    cells = [[_cell(row, column) for column in columns] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in cells)) if cells else len(header)
        for index, header in enumerate(headers)
    ]

    def line(values: Sequence[str]) -> str:
        return " ".join(
            f"{value:{column.align}{width}}" for value, width, column in zip(values, widths, columns)
        ).rstrip()

    print(line(headers))
    print("-" * sum(widths) + "-" * (len(widths) - 1))
    for row in cells:
        print(line(row))


def finish(rows: Sequence[RowResult], *, pass_message: str = "") -> int:
    """Print the final verdict for ``rows`` and return the process exit code."""
    failures = [row for row in rows if row.is_fail]
    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)}/{len(rows)} rows failed)")
        return 1
    if rows and all(row.is_skip for row in rows):
        reason = rows[0].detail or "environment unavailable"
        print(f"RESULT: SKIP ({reason})")
        return 0
    skipped = sum(row.is_skip for row in rows)
    note = pass_message or f"{len(rows)} rows"
    if skipped:
        note = f"{note}, {skipped} skipped"
    print(f"RESULT: PASS ({note})")
    return 0


def skip(reason: str, *, require_gpu: bool = False) -> int:
    """Report a job that cannot run on this machine (``--require-gpu`` makes it a failure)."""
    if require_gpu:
        print(f"RESULT: FAIL ({reason}; --require-gpu)")
        return 1
    print(f"RESULT: SKIP ({reason})")
    return 0
