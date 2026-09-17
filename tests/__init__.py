"""TyBoK regression checks.

Two levels, one entry point::

    cd TyBoK
    python tests/run_tests.py                          # every model, quick level
    python tests/run_tests.py --level full             # the pre-commit / CI regression

``tests/`` depends on the ``tybok`` package and torch only. See ``tests/README.md`` for the
layout, the levels, the child-process result protocol and why some checks exist for one model
only.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
