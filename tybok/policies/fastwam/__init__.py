"""fastWAM policy backend (registered as ``model_type="fastwam"``).

:class:`FastWAMEngine` is exported lazily (PEP 562): importing it pulls in the whole model stack
(torch, and triton through ``models/fp8_linear.py``), which is exactly what the pure-Python parts
of this package -- ``kernels/sweep.py``, ``geometry.py``, ``registry.py``, ``phases.py``, and
``config.py`` -- must not require. ``tests/fastwam/sweep_geom.py`` is the case that shows it: it
runs on a CPU-only runner (no triton), as do the documented ``python -m
tybok.policies.fastwam.kernels.sweep`` / ``...geometry`` / ``...registry`` entry points.

Registration does not go through this module either: :func:`tybok.policies.register_all` imports
``fastwam.engine`` by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # never imported at run time; see __getattr__ below
    from .engine import FastWAMEngine

__all__ = ["FastWAMEngine"]


def __getattr__(name: str):
    """Expose :class:`FastWAMEngine` without importing the model stack until it is used."""
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
