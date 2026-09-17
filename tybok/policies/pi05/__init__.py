"""pi0.5 policy backend (registered as ``model_type="pi05"``).

The engine and its validation report are exported lazily (PEP 562): importing them pulls in the
whole model stack, which ``pi05/config.py`` and the loader tooling must not require. Registration
does not go through this module either: :func:`tybok.policies.register_all` imports ``pi05.engine``
by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # never imported at run time; see __getattr__ below
    from .engine import PI05Engine, print_validation_report

__all__ = ["PI05Engine", "print_validation_report"]


def __getattr__(name: str):
    """Expose the engine exports without importing the model stack until they are used."""
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
