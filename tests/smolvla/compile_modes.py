"""smolvla: do the ``--compile`` combinations run, and do the regions really take effect?

smolvla compiles the phases it declares in ``eng.compiled_regions`` (``embed_prefix``,
``vlm_with_expert.forward``, ``denoise_step``). ``--compile`` forces ``--overlap`` off, and the
engine additionally disables the expert prefix-KV projection cache while compiling (it is
per-request mutable state that dynamo guards on) and restores it for the runtime path -- see
``tybok/policies/smolvla/engine.py``. The check reports the regions that really took effect, so a
silently ignored region fails instead of passing.

See :mod:`tests._common.compile_modes` for the shared driver, the modes and the verdict rules.

Usage::

    cd TyBoK
    python tests/smolvla/compile_modes.py
    python tests/smolvla/compile_modes.py --level full
    python tests/smolvla/compile_modes.py --list-jobs
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python tests/smolvla/compile_modes.py`` without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._common.compile_modes import CompileConfig, compile_modes_check  # noqa: E402
from tests.smolvla.spec import SMOLVLA  # noqa: E402

__all__ = ["main"]

CONFIG = CompileConfig(SMOLVLA)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smolvla compile-mode check (driver), or one job of it (``--job-index``)."""
    return compile_modes_check(CONFIG, Path(__file__), __doc__ or "", argv)


if __name__ == "__main__":
    sys.exit(main())
