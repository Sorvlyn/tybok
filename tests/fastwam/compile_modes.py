"""fastwam: do the ``--compile`` combinations run, and do the regions really take effect?

fastwam compiles *regions* rather than the whole core (``mot.prefill_video_cache`` and
``denoise_step``), and ``--compile`` forces ``--overlap`` off: with overlap the
compiled code enters ``torch.cuda.stream`` inside a loop and CUDA capture loses the fork/join
accounting (``cudaErrorStreamCaptureUnjoined``).
That is why the third row requests overlap and reports whether the engine kept it.

See :mod:`tests._common.compile_modes` for the shared driver, the modes and the verdict rules.

Usage::

    cd TyBoK
    python tests/fastwam/compile_modes.py
    python tests/fastwam/compile_modes.py --level full
    python tests/fastwam/compile_modes.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/fastwam/compile_modes.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.compile_modes import CompileConfig, compile_modes_check  # noqa: E402
from tests.fastwam.spec import FASTWAM  # noqa: E402

__all__ = ["main"]

CONFIG = CompileConfig(
    FASTWAM,
    # bf16/fp8 fastwam compiles need the text encoder off-GPU on a 16 GB card
    extra_kwargs={"text_encoder_device": "cpu"},
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fastwam compile-mode check (driver), or one job of it (``--job-index``)."""
    return compile_modes_check(CONFIG, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
