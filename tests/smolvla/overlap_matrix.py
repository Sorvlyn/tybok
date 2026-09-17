"""smolvla overlap matrix: (eager | graph) x (plain | fused).

See :mod:`tests._common.overlap_matrix` for the shared driver, the mode definitions and the
verdict rules; this module supplies the model and its history.

History (2026-09-14) -- what this gate caught on smolvla
--------------------------------------------------------
The step-0 expert cross-attention **silently fell back to the eager kernel** in the overlap
pipeline (2.71e-03 off): the post-prefill expert prefix-KV cache is not populated yet when the
step-0 layer runs on the side stream, so the fused path could not be used and the request quietly
took the eager branch. The fix projects the expert prefix K/V early enough; this matrix locks in
that every mode is bit-exact against the same mode with overlap off.

Usage::

    cd tybok
    python tests/smolvla/overlap_matrix.py
    python tests/smolvla/overlap_matrix.py --level full
    python tests/smolvla/overlap_matrix.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/smolvla/overlap_matrix.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.overlap_matrix import overlap_matrix_check  # noqa: E402
from tests.smolvla.spec import SMOLVLA  # noqa: E402 - after the path bootstrap

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smolvla overlap matrix (driver), or one job of it (``--job-index``)."""
    return overlap_matrix_check(SMOLVLA, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
