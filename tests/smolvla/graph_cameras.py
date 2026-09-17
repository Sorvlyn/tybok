"""smolvla: is ``--graph-cameras 2,3`` really giving working graphs for 2 *and* 3 images?

smolvla runs one ViT pass per camera slot, so the prefix length -- and therefore the captured
static buffers -- changes with the camera count: genuinely different graphs, one per count. That is
also why the graph key is ``(n_cams,)`` and nothing else: the language block is always padded to
``tokenizer_max_length``, so the count is the only dimension that can vary between requests.

This checkpoint has 3 camera keys, so both 2 and 3 are exercised. See
:mod:`tests._common.graph_cameras` for the shared driver and the verdict rules.

Usage::

    cd TyBoK
    python tests/smolvla/graph_cameras.py
    python tests/smolvla/graph_cameras.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/smolvla/graph_cameras.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.graph_cameras import CameraConfig, graph_cameras_check  # noqa: E402
from tests.smolvla.spec import SMOLVLA  # noqa: E402

__all__ = ["main"]

CONFIG = CameraConfig(SMOLVLA)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smolvla camera-count check (driver), or one job of it (``--job-index``)."""
    return graph_cameras_check(CONFIG, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
