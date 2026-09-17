"""pi05: do the ``--compile`` combinations run, and do the regions really take effect?

pi05 compiles the phases it declares in ``eng.compiled_regions`` (``embed_prefix``,
``paligemma_with_expert.forward``, ``denoise_step``). ``--compile`` is eager-only for pi05 and
forces ``--overlap`` off -- the check reports what the engine actually kept rather than what was
requested.

See :mod:`tests._common.compile_modes` for the shared driver, the modes and the verdict rules.

Usage::

    cd tybok
    python tests/pi05/compile_modes.py
    python tests/pi05/compile_modes.py --level full
    python tests/pi05/compile_modes.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/pi05/compile_modes.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.compile_modes import CompileConfig, compile_modes_check  # noqa: E402
from tests.pi05.spec import PI05  # noqa: E402

__all__ = ["main"]

CONFIG = CompileConfig(PI05)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pi05 compile-mode check (driver), or one job of it (``--job-index``)."""
    return compile_modes_check(CONFIG, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
