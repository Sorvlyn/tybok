"""Model registry: maps model types to :class:`~tybok.base.PolicyEngine`
subclasses so the CLI / worker / gateway can stay model-agnostic.

Each policy backend registers itself with :func:`register`; the checkpoint's
``config.json`` ``type`` field (lerobot convention, e.g. ``"smolvla"``) is used
to auto-detect the engine when ``--model-type`` is not given.

The backends are imported on first use (:func:`_ensure_backends`) rather than
when this module is imported: they pull in numpy / torch, and the CLI must stay
importable without the inference stack.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import PolicyEngine

_REGISTRY: dict[str, type[PolicyEngine]] = {}


def _ensure_backends() -> None:
    """Import the policy backends so they register themselves (idempotent)."""
    from .policies import register_all

    register_all()


def register(name: str):
    """Class decorator registering a :class:`PolicyEngine` under ``name``."""

    def register_class(cls: type[PolicyEngine]) -> type[PolicyEngine]:
        if name in _REGISTRY:
            raise ValueError(f"engine '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return register_class


def available() -> list[str]:
    """Registered backend names, i.e. :func:`shipped_models` loaded successfully.

    Imports every backend on first call (numpy / torch come with them); use
    :func:`shipped_models` when just the names are needed.
    """
    _ensure_backends()
    return sorted(_REGISTRY)


def shipped_models() -> list[str]:
    """Backend names this installation ships, without importing them.

    What ``python -m tybok models`` reports: it must answer on a machine where
    the inference stack is not installed (the same reason the backends are
    loaded lazily), and the answer is a property of the installation, not of
    whether the engines import. :func:`available` is the loaded counterpart.
    """
    from .policies import BACKENDS

    return sorted(BACKENDS)


def detect_model_type(checkpoint_dir: str) -> str:
    """Read the ``type`` field of a checkpoint's config.json (lerobot convention)."""
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mtype = cfg.get("type")
    if not mtype:
        raise ValueError(f"checkpoint {checkpoint_dir} has no 'type' field in config.json")
    return mtype


def _engine_kwargs(cls: type[PolicyEngine], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the engine constructor does not accept.

    Backends that declare ``**kwargs`` receive everything (they validate or
    ignore explicitly, e.g. fastwam's unsupported-flag blocklist); backends with
    a closed signature only get the names they declare, so the model-agnostic
    kwargs the worker always builds don't raise ``TypeError``.
    """
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    accepted = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


def reject_unsupported_flags(cls: type[PolicyEngine], model_type: str, kwargs: dict[str, Any]) -> None:
    """Raise when a flag this backend does not implement is explicitly set.

    Each engine may declare ``_UNSUPPORTED_FLAGS``: a mapping ``flag -> inert
    value``, where the inert value is what the model-agnostic worker kwargs carry
    by default (``False`` for an opt-in switch, ``True`` for a ``--no-*`` switch,
    ``None`` for an unset path). A passed value that differs from its inert
    default means the flag was requested; the backend does not implement it
    (it belongs to another model, or is a removed legacy flag), so fail loudly
    instead of silently ignoring it.

    Called by :func:`create_engine` on the *original* kwargs (before the
    closed-signature filtering in :func:`_engine_kwargs`), so it also covers
    backends like smolvla whose constructor never sees the unknown names.
    Backends that accept ``**kwargs`` may also call it from ``__init__`` to cover
    direct instantiation that bypasses :func:`create_engine`.
    """
    table = getattr(cls, "_UNSUPPORTED_FLAGS", None)
    if not table:
        return
    for flag, inert in table.items():
        if flag in kwargs and kwargs[flag] != inert:
            raise NotImplementedError(f"--{flag.replace('_', '-')} is not supported by the {model_type} backend")


def create_engine(
    checkpoint_dir: str,
    model_type: str | None = None,
    device: str = "auto",
    **kwargs: Any,
) -> PolicyEngine:
    """Instantiate a policy engine.

    Args:
        checkpoint_dir: path to the model checkpoint directory.
        model_type: registry key; auto-detected from ``config.json`` when None.
        device: torch device ("auto" / "cuda" / "cpu").
        **kwargs: forwarded to the engine constructor (filtered to the names its
            signature accepts, unless it declares ``**kwargs``). Flags a backend
            does not implement (its own removed legacy flags or another model's)
            raise ``NotImplementedError``.
    """
    if model_type is None:
        model_type = detect_model_type(checkpoint_dir)
    _ensure_backends()
    cls = _REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"unknown model type {model_type!r}; available: {available()} "
            f"(install the matching policy backend, e.g. `tybok.policies.smolvla`)"
        )
    reject_unsupported_flags(cls, model_type, kwargs)
    return cls(checkpoint_dir, device=device, **_engine_kwargs(cls, kwargs))
