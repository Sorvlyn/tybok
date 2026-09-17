"""The result protocol between a check's driver and its job processes.

A driver spawns itself once per job (see :mod:`tests._common.process`). The child prints exactly
one machine-readable line per job::

    ROW {"key": "graph | fused coop", "status": "PASS", "detail": "", "job": {...}, "metrics": {...}}

and, for humans and CI, one final ``RESULT: PASS|FAIL|SKIP`` line. ``job`` is the identity of
what ran (so a driver can pair a job with its reference process); ``metrics`` holds the numbers
the driver tabulates; ``values`` is the optional raw output vector for verdicts that can only be
decided by comparing two processes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ROW_PREFIX",
    "RESULT_PREFIX",
    "RowResult",
    "RowStatus",
    "emit_row",
    "parse_result_line",
    "parse_row_line",
]

ROW_PREFIX = "ROW "
RESULT_PREFIX = "RESULT:"


class RowStatus(str, Enum):
    """Verdict of one job."""

    PASS = "PASS"
    FAIL = "FAIL"
    #: The job could not run (no CUDA / no checkpoint) -- not a failure, exit code stays 0.
    SKIP = "SKIP"


@dataclass(frozen=True)
class RowResult:
    """One job's result, as reported by the child and consumed by the driver."""

    key: str
    status: RowStatus
    detail: str = ""
    job: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    values: Sequence[float] | None = None

    @property
    def is_pass(self) -> bool:
        return self.status is RowStatus.PASS

    @property
    def is_fail(self) -> bool:
        return self.status is RowStatus.FAIL

    @property
    def is_skip(self) -> bool:
        return self.status is RowStatus.SKIP

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the row (``values`` is dropped when unset)."""
        payload: dict[str, Any] = {
            "key": self.key,
            "status": self.status.value,
            "detail": self.detail,
            "job": dict(self.job),
            "metrics": dict(self.metrics),
        }
        if self.values is not None:
            payload["values"] = list(self.values)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RowResult":
        """Inverse of :meth:`to_dict`."""
        return cls(
            key=str(payload["key"]),
            status=RowStatus(payload["status"]),
            detail=str(payload.get("detail", "")),
            job=dict(payload.get("job", {})),
            metrics=dict(payload.get("metrics", {})),
            values=payload.get("values"),
        )

    def with_status(self, status: RowStatus, *, detail: str = "") -> "RowResult":
        """Same row, re-judged by the driver (e.g. after comparing two processes)."""
        return RowResult(
            key=self.key,
            status=status,
            detail=detail or self.detail,
            job=self.job,
            metrics=self.metrics,
            values=self.values,
        )


def emit_row(row: RowResult) -> None:
    """Print one row on the ``ROW`` line the driver parses."""
    print(ROW_PREFIX + json.dumps(row.to_dict()), flush=True)


def parse_row_line(line: str) -> RowResult | None:
    """Parse ``line`` if it is a ``ROW`` line, else return ``None``."""
    if not line.startswith(ROW_PREFIX):
        return None
    return RowResult.from_dict(json.loads(line[len(ROW_PREFIX) :]))


def parse_result_line(line: str) -> RowStatus | None:
    """Verdict of a ``RESULT: PASS|FAIL|SKIP`` line, or ``None`` if it is not one."""
    if not line.startswith(RESULT_PREFIX):
        return None
    verdict = line[len(RESULT_PREFIX) :].strip().split(" ", 1)[0]
    try:
        return RowStatus(verdict)
    except ValueError:
        return None
