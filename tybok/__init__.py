"""Policy deployment engine (multi-model).

Model backends live under :mod:`tybok.policies` and register themselves in
:mod:`tybok.registry`: currently ``fastwam``, ``pi05`` and ``smolvla``.

Components:

- ``tybok.worker``  : persistent inference process (owns the model)
- ``tybok.gateway`` : aiohttp WebSocket gateway + image processing
- ``tybok.policies``: model-specific engines

(Server-side package; the example WebSocket client lives in ``examples/client.py``.)

Importing this package is cheap: the backends are not loaded here but on first use of
:func:`tybok.registry.available` / :func:`tybok.registry.create_engine` (loading them imports
numpy / torch), so ``python -m tybok --help`` / ``models`` runs without the inference stack.
"""

__version__ = "0.1.0"

__all__ = ["policies"]


def __getattr__(name: str):
    """Expose :mod:`tybok.policies` without importing it with the package.

    ``import tybok; tybok.policies`` keeps working; it just no longer costs a
    numpy / torch import until it is actually used. Registration still happens
    through :func:`tybok.registry.available` / :func:`tybok.registry.create_engine`.
    """
    if name == "policies":
        from . import policies

        return policies
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
