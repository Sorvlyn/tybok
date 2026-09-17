"""``quick`` / ``full``: how much of a check to run.

``QUICK`` is what you run after every edit -- a subset of the jobs, fewer replays, minutes not
tens of minutes. ``FULL`` is the regression run (pre-commit / CI): every job, the production
replay counts. Both levels assert the *same* invariants (bit-exactness / overlap engagement);
the level only changes how many configurations and how many replays are exercised, so a green
``quick`` is evidence about the configurations it ran and nothing more.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, TypeVar

__all__ = ["Level", "Job", "jobs_for_level"]


class Level(str, Enum):
    """Test level: a cheap smoke gate or the full regression."""

    QUICK = "quick"
    FULL = "full"


class Job(Protocol):
    """A job declares its display key and whether the quick level covers it.

    Both are read-only here so a dataclass field and a ``@property`` both satisfy the protocol.
    """

    @property
    def key(self) -> str: ...

    @property
    def quick(self) -> bool: ...


JobT = TypeVar("JobT", bound=Job)


def jobs_for_level(jobs: Sequence[JobT], level: Level) -> list[JobT]:
    """Return the jobs of ``jobs`` that belong to ``level`` (``FULL`` runs all of them).

    Jobs keep their declared order, so a driver can pair a job with its reference by position
    (e.g. ``overlap on`` followed by ``overlap off``).
    """
    if level is Level.FULL:
        return list(jobs)
    return [job for job in jobs if job.quick]
