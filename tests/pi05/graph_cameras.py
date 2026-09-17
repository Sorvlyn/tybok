"""pi05: is ``--graph-cameras 2,3`` really giving working graphs for 2 *and* 3 images?

pi05 runs one ViT pass per camera slot, so the prefix length -- and therefore the captured static
buffers -- changes with the camera count: genuinely different graphs, one per count.

History (2026-09-14): this row used to be a drift row. The eager path truncated the language to
the exact live length and kept the always-empty camera placeholder slot, while the graph path
padded the language to the 16-multiple bucket and dropped that slot (~1.7e-3, plus a different
prefix length). Both paths now go through the same two rules (``PI05Config.live_camera_keys`` and
``PI05FlowMatching.pack_language``), so the row is strict again -- a non-zero value here means the
two paths drifted apart again.

The checkpoint exposes 2 camera keys, so the counts are clamped to 2 (which is the clamping the
engine documents). See :mod:`tests._common.graph_cameras` for the shared driver.

Usage::

    cd tybok
    python tests/pi05/graph_cameras.py
    python tests/pi05/graph_cameras.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/pi05/graph_cameras.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.graph_cameras import CameraConfig, graph_cameras_check  # noqa: E402
from tests.pi05.spec import PI05  # noqa: E402

__all__ = ["main"]

CONFIG = CameraConfig(
    PI05,
    # --pad-free makes the camera count the varying dimension (see PI05Config.live_camera_keys)
    extra_kwargs={"pad_free": True},
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pi05 camera-count check (driver), or one job of it (``--job-index``)."""
    return graph_cameras_check(CONFIG, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
