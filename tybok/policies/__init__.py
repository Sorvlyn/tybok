"""Model-specific policy backends.

Each subpackage implements a :class:`~tybok.base.PolicyEngine` and
registers it in :mod:`tybok.registry` via the ``@register`` decorator.
:func:`register_all` imports every backend, which is what performs the
registration.

Adding a new model (e.g. pi0.5, lingbot-vla):

1. create ``tybok/policies/<name>/`` with a ``PolicyEngine`` subclass
   decorated with ``@register("<name>")``
2. add ``<name>`` to ``BACKENDS`` so :func:`register_all` imports it

Registration is deliberately *not* a package-import side effect: importing a
backend pulls in numpy / torch (and, for the engines, the whole model stack),
while ``python -m tybok --help`` / ``python -m tybok models`` and the CLI
surface guard must work on a machine without the inference stack installed.
:mod:`tybok.registry` calls :func:`register_all` on first use instead.
"""

import importlib

# Backend package names this installation ships, in report order. The package directory name
# is the registry key the engine declares with ``@register`` (``<name>/engine.py``).
BACKENDS = ("fastwam", "pi05", "smolvla")


def register_all() -> None:
    """Import every backend so its engine registers itself.

    Idempotent: ``importlib`` returns the cached module, and the ``@register``
    decorator only runs on the first import of a backend.
    """
    for name in BACKENDS:
        importlib.import_module(f"{__name__}.{name}")


__all__ = ["BACKENDS", "register_all"]
