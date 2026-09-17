"""Model-specific policy backends.

Each subpackage implements a :class:`~tybok.base.PolicyEngine` and
registers it in :mod:`tybok.registry` via the ``@register`` decorator.
:func:`register_all` imports every backend's engine, which is what performs the
registration.

Adding a new model (e.g. pi0.5, lingbot-vla):

1. create ``tybok/policies/<name>/`` with a ``PolicyEngine`` subclass
   decorated with ``@register("<name>")``
2. add ``<name>`` to ``BACKENDS`` so :func:`register_all` imports its engine

Registration is deliberately *not* a package-import side effect: importing an
engine pulls in numpy / torch (for the engines, the whole model stack), while
``python -m tybok --help`` / ``python -m tybok models`` and the CLI surface
guard must work on a machine without the inference stack installed.
:mod:`tybok.registry` calls :func:`register_all` on first use instead.

For the same reason :func:`register_all` imports ``<name>.engine`` rather than
the ``<name>`` package, and each backend package exposes its engine lazily (PEP
562): a backend subpackage that needs no model stack -- fastwam's ``kernels/``
sweep / geometry / registry tooling, the backends' ``config`` -- stays
importable where that stack (torch, triton) is not installed.
"""

import importlib

# Backend package names this installation ships, in report order. The directory name is the
# registry key the engine declares with ``@register`` (``<name>/engine.py``); ``register_all``
# imports exactly that module.
BACKENDS = ("fastwam", "pi05", "smolvla")


def register_all() -> None:
    """Import every backend's ``engine`` module so it registers itself.

    The module, not the package: importing the package would run its eager
    exports, and a backend package must stay importable without the model stack
    (see the module docstring).

    Idempotent: ``importlib`` returns the cached module, and the ``@register``
    decorator only runs on the first import of a backend.
    """
    for name in BACKENDS:
        importlib.import_module(f"{__name__}.{name}.engine")


__all__ = ["BACKENDS", "register_all"]
