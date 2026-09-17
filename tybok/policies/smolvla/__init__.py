"""SmolVLA policy backend.

Deployment engine (SigLIP vision encoder + Gemma text decoder + action expert +
flow-matching denoising), registered as ``model_type="smolvla"``.

``engine.py`` (engine) / ``config.py`` (checkpoint + VLM config) /
``tokenizer.py`` (task tokenization) / ``preprocess.py`` (pre/post + stats) /
``models/`` (PyTorch ``nn.Module`` code).

The engine and its validation report are exported lazily (PEP 562): importing them pulls in the
whole model stack, which ``smolvla/config.py`` must not require. Registration does not go through
this module either: :func:`tybok.policies.register_all` imports ``smolvla.engine`` by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # never imported at run time; see __getattr__ below
    from .engine import SmolVLAEngine, print_validation_report

__all__ = ["SmolVLAEngine", "print_validation_report"]


def __getattr__(name: str):
    """Expose the engine exports without importing the model stack until they are used."""
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
