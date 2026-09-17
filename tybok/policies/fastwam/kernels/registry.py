"""Capability contract and dispatch plan for fused kernels (query only; does not change engine behavior).

Each platform kernel registers a :class:`KernelSpec`: compile-time geometry, verified archs,
whether cross-CTA reductions are order-dependent, the numerics tier, and the forms it provides:

* ``split_phases > 0`` -- non-cooperative split-phase form: one plain launch per phase, no
  co-residency constraint, can run in multiple waves, works on small cards / shared cards / MPS
  partitions; this is the portable form.
* ``coop`` -- cooperative form: a single ``cudaLaunchCooperativeKernel`` + ``grid.sync()``,
  requiring the whole grid to be resident at once (fast, but saturates the device and cannot
  coexist with other tasks). It is an acceleration, not a requirement.

Three rules (blocking pitfalls this repo has hit):

1. **A kernel that only provides the cooperative form must not be registered** (``coop and
   split_phases <= 0`` errors immediately): otherwise there is no path on cards that cannot
   achieve co-residency.
2. **Structural lower bounds must be declared explicitly** (``min_grid``): in constructs like
   ``if (b < K)`` / claim-style row loops
   (``for (r = blockIdx.x; r < M; r += gridDim.x)`` where K is set by the problem size), a grid
   below the lower bound is an **undercount** (silently wrong), not just slower; the dispatcher
   computes and validates the grid accordingly.
3. **``reduction`` marks whether cross-CTA reductions are order-dependent**: with ``"atomic"``
   the result varies with execution order (not reproducible for identical inputs); ``"fixed"``
   is a fixed order, and the latter is the admission requirement.

Usage (diagnostics): ``python -m tybok.policies.fastwam.kernels.registry``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Archs these kernels have been verified on: **not** a restriction, just a record of "actually
# ran on this"; unlisted = "not verified", not "definitely won't work" (probe() says so honestly).
VERIFIED_ARCHS: tuple[tuple[int, int], ...] = ((8, 9),)

_LOADERS: dict[str, Callable[[], Any]] = {}


def _loader(kind: str):
    """Import and compile the loader for the corresponding extension on demand (first call triggers compilation)."""
    if not _LOADERS:
        from . import load_action_pre_ext, load_fused_ext, load_tmt5_fused_ext, load_video_fused_ext

        _LOADERS.update(
            action=load_fused_ext, video=load_video_fused_ext, tmt5=load_tmt5_fused_ext, adit_pre=load_action_pre_ext
        )
    return _LOADERS[kind]


@dataclass(frozen=True)
class KernelSpec:
    """Capability contract of one fused kernel (the `*_grid` / `*_coop_grid` functions are exposed by the extension's bindings)."""

    # fmt: off
    name: str                     # "adit.attn_self"
    ext: str                      # "action" | "video" | "tmt5" | "adit_pre"
    split_phases: int             # >0 = provides the non-cooperative split-phase form
    coop: bool                    # whether a cooperative form is also provided
    min_grid: int                 # structural lower bound (below it = undercount)
    smem_bytes: int | None        # dynamic smem; None = decided at runtime (follows context length)
    geometry: dict[str, int] = field(default_factory=dict)
    archs: tuple[tuple[int, int], ...] = VERIFIED_ARCHS
    reduction: str = "fixed"      # "fixed" | "atomic"
    numerics: str = "drift"       # "bitwise" | "drift" | "bf16"
    grid_args: tuple = ()         # runtime arguments needed when querying grid (e.g. B / L)
    split_grid_fn: str | None = None
    coop_grid_fn: str | None = None
    grid_mode: str = "problem"    # "problem" (set by problem size) | "capacity" (occupancy x SM)
    # fmt: on


_REGISTRY: list[KernelSpec] = []


def register(spec: KernelSpec) -> KernelSpec:
    """Register a kernel; the contract is validated **at registration time** and violations raise, never deferred to runtime.

    Raises:
        ValueError: when the cooperative form is inconsistent with ``split_phases`` / ``coop_grid_fn``.
    """
    if spec.coop and spec.split_phases <= 0:
        raise ValueError(
            f"{spec.name}: providing the cooperative form requires also providing the "
            f"non-cooperative split-phase form (split_phases > 0). A kernel relying only on the "
            f"cooperative form has no path on small/shared cards."
        )
    if spec.coop and spec.coop_grid_fn is None:
        raise ValueError(f"{spec.name}: coop=True but coop_grid_fn not given, cannot query availability")
    if not spec.coop and spec.coop_grid_fn is not None:
        raise ValueError(f"{spec.name}: coop=False but coop_grid_fn was given")
    _REGISTRY.append(spec)
    return spec


def specs() -> tuple[KernelSpec, ...]:
    return tuple(_REGISTRY)


def spec(name: str) -> KernelSpec:
    for s in _REGISTRY:
        if s.name == name:
            return s
    raise KeyError(f"unregistered kernel {name} (available: {[s.name for s in _REGISTRY]})")


# --------------------------------------------------------------------------- #
# Built-in registry (values taken from each .cu's compile-time constants)
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="adit.attn_self",
        ext="action",
        split_phases=6,
        coop=True,
        min_grid=96,
        smem_bytes=0,
        grid_mode="problem",
        split_grid_fn="attn_self_grid",
        coop_grid_fn="attn_self_coop_grid",
        geometry=dict(M=32, H=1024, H3=3072, F=4096, heads=24, head_dim=128),
        numerics="drift",
    )
)
register(
    KernelSpec(
        name="adit.attn_cross",
        ext="action",
        split_phases=6,
        coop=True,
        min_grid=48,
        smem_bytes=None,  # dynamic smem = L-byte mask (at runtime)
        grid_mode="problem",
        grid_args=("L",),
        split_grid_fn="adit_attn_cross_grid",
        coop_grid_fn="adit_attn_cross_coop_grid",
        geometry=dict(M=32, H=1024, NQ=3072, NO=1024, context=129),
        numerics="drift",
    )
)
register(
    KernelSpec(
        name="adit.ffn",
        ext="action",
        split_phases=4,
        coop=True,
        min_grid=64,
        smem_bytes=0,
        grid_mode="problem",
        grid_args=("B",),
        split_grid_fn="ffn_grid",
        coop_grid_fn="ffn_coop_grid",
        geometry=dict(M=32, H=1024, F=4096, GRID0=64),
        numerics="drift",
    )
)
register(
    KernelSpec(
        name="vdit.attn_self",
        ext="video",
        split_phases=6,
        coop=True,
        min_grid=120,
        smem_bytes=49152,
        grid_mode="capacity",
        split_grid_fn="attn_self_split_grid",
        coop_grid_fn="attn_self_grid",
        geometry=dict(M=120, H=3072, N=9216, heads=24, head_dim=128),
        numerics="drift",
    )
)
register(
    KernelSpec(
        name="vdit.attn_cross",
        ext="video",
        split_phases=6,
        coop=True,
        min_grid=120,
        smem_bytes=49152,
        grid_mode="capacity",
        split_grid_fn="attn_cross_split_grid",
        coop_grid_fn="attn_cross_grid",
        geometry=dict(M=120, C=129, H=3072, heads=24, head_dim=128),
        numerics="drift",
    )
)
register(
    KernelSpec(
        name="vdit.ffn",
        ext="video",
        split_phases=4,
        coop=True,
        min_grid=120,
        smem_bytes=49152,
        grid_mode="capacity",
        split_grid_fn="ffn_split_grid",
        coop_grid_fn="ffn_grid",
        geometry=dict(M=120, H=3072, F=14336),
        numerics="drift",
    )
)
register(
    KernelSpec(
        name="tmt5.ffn",
        ext="tmt5",
        split_phases=4,
        coop=True,
        min_grid=1,
        smem_bytes=98304,
        grid_mode="capacity",
        split_grid_fn="ffn_grid",
        coop_grid_fn="ffn_coop_grid",
        geometry=dict(M=128, H=4096, F=10240),
        numerics="bitwise",  # except F_PHASE_1's RMSNorm reduction order
    )
)
register(
    KernelSpec(
        name="tmt5.attn",
        ext="tmt5",
        split_phases=5,
        coop=True,
        min_grid=1,
        smem_bytes=98304,
        grid_mode="capacity",
        split_grid_fn="attn_grid",
        coop_grid_fn="attn_coop_grid",
        geometry=dict(M=128, H=4096, N=12288, heads=64, head_dim=64),
        numerics="bitwise",  # except S_PHASE_1's row reduction order
    )
)
register(
    KernelSpec(
        name="adit_pre",
        ext="adit_pre",
        split_phases=0,
        coop=False,
        min_grid=1,
        smem_bytes=None,
        geometry=dict(H=1024, OUT=6144),
        numerics="bitwise",  # the time path differs in fp32 reduction order
    )
)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Availability:
    """Availability of a kernel on **this device**. When form == "none", reason explains why."""

    # fmt: off
    spec: KernelSpec
    form: str                     # "coop" | "split" | "none"
    grid: int | None
    reason: str = ""              # "" = the preferred form is available; otherwise the reason for downgrade/unavailability
    # fmt: on


def _cap(device: Any = None) -> tuple[int, int]:
    import torch

    return tuple(torch.cuda.get_device_capability(device))


def coop_available(spec: KernelSpec, device: Any = None) -> tuple[bool, str]:
    """Ask the device only: whether the kernel's cooperative form can currently run (regardless of whether the arch was "verified").

    For engine downgrade: on cards/partitions that cannot achieve co-residency,
    ``--cooperative-kernel`` should fall back to split phases rather than blowing up on the
    first launch. Triggers extension compilation (which is needed soon anyway).

    Returns:
        ``(available, reason)``; the reason is an empty string when available.
    """
    if not spec.coop:
        return False, f"{spec.name} does not provide the cooperative form"
    try:
        ext = _loader(spec.ext)()
        args = tuple(_resolve_arg(spec, k) for k in spec.grid_args)
        int(getattr(ext, spec.coop_grid_fn)(*args))
        return True, ""
    except RuntimeError as exc:  # insufficient capacity / cooperative launch unsupported / query failed
        return False, str(exc).splitlines()[0]
    except Exception as exc:  # noqa: BLE001 - build failures etc. must also allow downgrade
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def probe(spec: KernelSpec, device: Any = None) -> Availability:
    """Pure query: no launch, no behavior change; if the cooperative form fails, fall back to split phases and record the reason.

    Being able to "ask first" is because the failure paths in the kernels were changed from
    ``exit(1)`` to raising ``RuntimeError`` -- previously a query would kill the process.
    """
    cap = _cap(device)
    if cap not in spec.archs:
        return Availability(
            spec,
            "none",
            None,
            f"not built/verified on sm_{cap[0]}{cap[1]} (verified: "
            + ", ".join(f"sm_{a}{b}" for a, b in spec.archs)
            + ")",
        )
    ext = _loader(spec.ext)()
    args = tuple(_resolve_arg(spec, k) for k in spec.grid_args)

    coop_reason = ""
    if spec.coop:
        try:
            return Availability(spec, "coop", int(getattr(ext, spec.coop_grid_fn)(*args)))
        except RuntimeError as exc:  # insufficient capacity / cooperative launch unsupported
            coop_reason = str(exc)
    if spec.split_grid_fn is None:
        # A component that only does plain launch (adit_pre): no grid contract to query, naturally portable.
        return Availability(spec, "split", None, coop_reason)
    return Availability(spec, "split", int(getattr(ext, spec.split_grid_fn)(*args)), coop_reason)


def _resolve_arg(spec: KernelSpec, key: str) -> int:
    """Resolve the runtime arguments needed for a grid query; by default takes the constant value from the model config."""
    if key == "B":
        return 1  # one instance of the action expert at a time
    if key == "L":
        return spec.geometry["context"]
    raise KeyError(key)


def kernel_plan(device: Any = None, *, kinds: tuple[str, ...] | None = None) -> list[Availability]:
    """Query the whole table. ``kinds`` can restrict to a few extensions, avoiding compiling extensions for tiers that are not enabled."""
    return [probe(s, device) for s in _REGISTRY if kinds is None or s.ext in kinds]


def format_plan(plan: list[Availability]) -> str:
    import torch

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    cap = _cap(dev)
    lines = [f"kernel plan (sm_{cap[0]}{cap[1]}, {props.multi_processor_count} SM, {props.total_memory // 2**30} GB):"]
    for av in plan:
        s = av.spec
        grid = "n/a" if av.grid is None else str(av.grid)
        phases = f"{s.split_phases} phases" if s.split_phases else "plain launch"
        lines.append(
            f"  {s.name:<16} {s.ext:<8} {av.form:<6} grid={grid:<5} min={s.min_grid:<4} "
            f"{phases:<9} reduction={s.reduction:<6} {s.numerics}"
        )
        if av.reason:
            lines.append(f"      -> {av.reason.splitlines()[0]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_plan(kernel_plan()))
