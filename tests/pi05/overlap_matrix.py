"""pi05 overlap matrix: (eager | graph) x (plain | fused).

See :mod:`tests._common.overlap_matrix` for the shared driver, the mode definitions and the
verdict rules; this module supplies the model and its history.

History (2026-09-14) -- what this gate caught on pi05
-----------------------------------------------------
``_prefix_kT_interleaved`` cached an **activation** layout keyed by ``data_ptr()``. During
capture the buffer is reused, so the pack ops that feed the cross-attention were never recorded
into the per-layer graphs (the warmup-time prefix keys were cached instead) and every replay
cross-attended against stale keys -- 2.92e-01 off, in graph mode only. The fix recomputes the
interleaved keys every request; this matrix locks in that every mode is bit-exact against the same
mode with overlap off.

Usage::

    cd TyBoK
    python tests/pi05/overlap_matrix.py
    python tests/pi05/overlap_matrix.py --level full
    python tests/pi05/overlap_matrix.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/pi05/overlap_matrix.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.overlap_matrix import overlap_matrix_check  # noqa: E402
from tests.pi05.spec import PI05  # noqa: E402 - after the path bootstrap

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pi05 overlap matrix (driver), or one job of it (``--job-index``)."""
    return overlap_matrix_check(PI05, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
