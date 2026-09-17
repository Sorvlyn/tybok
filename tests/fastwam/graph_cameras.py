"""fastwam: is ``--graph-cameras 2,3`` accepted, and does the engine still run?

fastwam has **one** graph shape for every camera count: the preprocessor concatenates the camera
images into a single frame and the VAE runs outside the captured core, so the count never enters
the graph key. The engine logs ``--graph-cameras`` as a no-op for this backend; this check asserts
the flag is accepted, that a graph is still captured, and that every camera count the checkpoint
supports stays bit-exact against the eager path.

See :mod:`tests._common.graph_cameras` for the shared driver and the verdict rules.

Usage::

    cd TyBoK
    python tests/fastwam/graph_cameras.py
    python tests/fastwam/graph_cameras.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/fastwam/graph_cameras.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.graph_cameras import CameraConfig, graph_cameras_check  # noqa: E402
from tests.fastwam.spec import FASTWAM  # noqa: E402

__all__ = ["main"]

CONFIG = CameraConfig(FASTWAM)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fastwam camera-count check (driver), or one job of it (``--job-index``)."""
    return graph_cameras_check(CONFIG, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
