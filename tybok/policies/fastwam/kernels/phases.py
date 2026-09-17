"""Phase-operator registry: **the registration unit = a phase**.

Each phase of a kernel is an independently launchable operator (phases interact only through
globals), so it can be replaced phase by phase: the host engine's operators can take over
`{S_PHASE_1, S_PHASE_2}` while the remaining phases keep using the default implementation (the
existing tuned kernels).

Three levels of use: L1 runs without touching parameters (default implementation = the phase
bodies of the existing kernels), L2 sweeps geometry (`KernelIO.geometry` / `PhaseSpec.geometry`
are template parameters), L3 registers a phase with `replace("tmt5.ffn/P3", fn)`.

Why the ABI uses **named tensors** instead of positional parameters: a host operator cannot use
20 raw pointers, so the positional-parameter order is moved into the registry (which is contract
data anyway) and restored into the ext's positional call by `_default_call`.

`validate()` compares the declared parameter order one by one against the signature pybind
writes into `__doc__` -- a wrong order, a missing parameter, or an extra parameter is reported
on the spot instead of becoming a wrong answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# --------------------------------------------------------------------------- #
# Contract data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Buffer:
    """A tensor parameter: the name matches the ext's positional parameter name (= the name in the contract)."""

    # fmt: off
    shape: str          # human-readable ("MxH"); the more accurate it is, the easier L3 is to write
    dtype: str          # bf16 / fp8 / fp32 / u32 / u8
    note: str = ""
    # fmt: on


@dataclass(frozen=True)
class Numerics:
    """Numerics contract for the math a phase **does itself**. Items it does not do are left None.

    A replacement implementation must fill in every item this phase is involved in
    (`validate()` judges by `acceptance`).
    """

    # fmt: off
    norm_kind: str | None = None          # rms | layer_affine | layer_plain
    norm_eps: float | None = None         # note: must be 1/sqrtf, not rsqrtf
    determinism: str | None = None        # fixed | atomic (cross-CTA reductions must be fixed)
    reduction_notes: str | None = None    # parallel shape of the reduction (how cross-CTA is combined, who writes and who reads)
    rounding_points: tuple[str, ...] = () # position of each cast in this phase, listed one by one
    fp8_recipe: str | None = None         # amax/448 + software RNE + true division
    input_precision: str | None = None    # bf16 | fp32_preserved
    residual: str | None = None           # single_round | double_round
    activation: str | None = None         # **the literal formula**, not just "GELU"
    accum_width: str | None = None        # fp32 (fp16 forbidden)
    divide: str | None = None             # div.rn (IEEE) | div.full (non-IEEE)
    attention: str | None = None
    rope: str | None = None
    # fmt: on


@dataclass(frozen=True)
class PhaseSpec:
    """The registration unit. `name = "<kernel>/P<index>"`."""

    # fmt: off
    kernel: str
    index: int                            # = only_phase number (1..N)
    does: str                             # one sentence
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    numerics: Numerics = field(default_factory=Numerics)
    acceptance: str = "bitwise"           # bitwise | drift -- acceptance tier when replacing
    tiles: tuple[str, ...] = ()           # GEMM tile labels used to instantiate this phase (= keys of
                                          # the geometry table, also the `fwam_tiles("<label>", ...)`
                                          # self-reported name); sweeps use it to go from "phase" to "macro".
    # fmt: on

    @property
    def name(self) -> str:
        return f"{self.kernel}/P{self.index}"


# --------------------------------------------------------------------------- #
# Kernel IO: positional parameter order (**copied from bindings, reconciled against the pybind
# signature by validate()**)
# --------------------------------------------------------------------------- #
# Three kinds of parameters: tensor t / scalar s / injected p (only_phase number) and r (stream)
_Arg = tuple[str, str]


@dataclass(frozen=True)
class KernelIO:
    # fmt: off
    kernel: str                # "tmt5.ffn"
    ext: str                   # extension name (loader key in kernels/__init__.py)
    ext_fn: str                # function name inside the ext
    args: tuple[_Arg, ...]     # ordered positional parameters
    buffers: Mapping[str, Buffer]
    geometry: Mapping[str, int] = field(default_factory=dict)
    grid_policy: str = "capacity"      # capacity | problem
    min_grid: int = 1
    coop: bool = True                  # whether a cooperative form also exists (T2) -- not part of the phase registry
    phases: tuple[int, ...] = ()       # list of phase numbers (1..N)
    # fmt: on

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(n for k, n in self.args if k == "t")


KERNEL_IO: dict[str, KernelIO] = {}


def register_kernel(io: KernelIO) -> KernelIO:
    if io.kernel in KERNEL_IO:
        raise ValueError(f"kernel {io.kernel} registered twice")
    declared = set(n for k, n in io.args if k == "t")
    missing = declared - set(io.buffers)
    extra = set(io.buffers) - declared
    if missing or extra:
        raise ValueError(
            f"{io.kernel}: buffers and positional parameters disagree "
            f"(missing {sorted(missing)} / extra {sorted(extra)})"
        )
    if io.phases != tuple(range(1, len(io.phases) + 1)):
        raise ValueError(f"{io.kernel}: phases must be 1..N, got {io.phases}")
    KERNEL_IO[io.kernel] = io
    return io


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
PHASES: dict[str, PhaseSpec] = {}


def register_phase(spec: PhaseSpec) -> PhaseSpec:
    if spec.kernel not in KERNEL_IO:
        raise ValueError(f"{spec.name}: kernel {spec.kernel} is not registered")
    io = KERNEL_IO[spec.kernel]
    if spec.index not in io.phases:
        raise ValueError(f"{spec.name}: phase number {spec.index} is not in {io.phases}")
    known = set(io.buffers)
    for side, names in (("reads", spec.reads), ("writes", spec.writes)):
        unknown = set(names) - known
        if unknown:
            raise ValueError(f"{spec.name}: {side} contains undeclared buffer {sorted(unknown)}")
    if spec.name in PHASES:
        raise ValueError(f"{spec.name}: registered twice")
    PHASES[spec.name] = spec
    return spec


def phases_of(kernel: str) -> tuple[PhaseSpec, ...]:
    return tuple(PHASES[f"{kernel}/P{i}"] for i in KERNEL_IO[kernel].phases)


# --------------------------------------------------------------------------- #
# Execution: default implementation = restore the ext's positional call from named tensors
# --------------------------------------------------------------------------- #
_EXT_CACHE: dict[str, Any] = {}
_REPLACED: dict[str, Callable[..., None]] = {}
# The currently active recorder (`phase_check.Recorder`; None = no recording). Kept here rather
# than in phase_check so dispatch need not import phase_check (avoids a circular import and extra overhead).
_RECORDER: Any = None


def _ext(name: str) -> Any:
    if name not in _EXT_CACHE:
        from . import load_action_pre_ext, load_fused_ext, load_tmt5_fused_ext, load_video_fused_ext

        _EXT_CACHE.update(
            action=load_fused_ext(),
            video=load_video_fused_ext(),
            tmt5=load_tmt5_fused_ext(),
            adit_pre=load_action_pre_ext(),
        )
    return _EXT_CACHE[name]


def _build_args(io: KernelIO, index: int, tensors: Mapping[str, Any], scalars: Mapping[str, Any], stream: int) -> list:
    pos = []
    for kind, name in io.args:
        if kind == "t":
            pos.append(tensors[name].data_ptr())
        elif kind == "s":
            pos.append(scalars[name])
        elif kind == "p":
            pos.append(index)
        elif kind == "r":
            pos.append(stream)
        else:  # pragma: no cover
            raise AssertionError(f"unknown parameter kind {kind!r}")
    return pos


def default_call(
    kernel: str, index: int, *, tensors: Mapping[str, Any], scalars: Mapping[str, Any], stream: int
) -> None:
    """Default implementation (= the existing tuned kernels). A replacement implementation may call it to forward/instrument."""
    io = KERNEL_IO[kernel]
    getattr(_ext(io.ext), io.ext_fn)(*_build_args(io, index, tensors, scalars, stream))


def dispatch(kernel: str, index: int, *, tensors: Mapping[str, Any], scalars: Mapping[str, Any], stream: int) -> None:
    """Run one phase. `index < 0` = cooperative full pipeline (the T2 form, not a phase, bypasses the registry).

    The default implementation is **identical word for word** to before this file was introduced (same chain of positional parameters, same ext call).

    WARNING: **replacement only takes effect in the non-cooperative (split-phase) form**: the
    cooperative form runs the whole chain in a single `cudaLaunchCooperativeKernel`
    (`index = -1`), leaving no "phase" to replace -- which is exactly the necessary consequence
    of "host-operator integration = split-phase form".
    """
    rec = _RECORDER
    if rec is not None and index >= 0:
        rec.enter(kernel, index, tensors, scalars)
    if index >= 0:
        fn = _REPLACED.get(f"{kernel}/P{index}")
        if fn is not None:
            fn(tensors=tensors, scalars=scalars, stream=stream)
            if rec is not None:
                rec.exit(kernel, index, tensors)
            return
    default_call(kernel, index, tensors=tensors, scalars=scalars, stream=stream)
    if rec is not None and index >= 0:
        rec.exit(kernel, index, tensors)


def replace(phase_name: str, fn: Callable[..., None] | None) -> None:
    """Let an implementation take over a phase (`fn=None` = revert to the default implementation).

    `fn(*, tensors, scalars, stream)`; see the phase's `PhaseSpec` for the contract. Before
    replacing, first run `validate()` (contract completeness) and per-phase bit comparison.

    **Only takes effect in the split-phase form** (the cooperative form has no phases, see
    `dispatch`). To forward to the default implementation, call
    `default_call(kernel, index, ...)`.
    """
    if fn is not None:
        spec = PHASES.get(phase_name)
        if spec is None:
            raise KeyError(f"unregistered phase {phase_name}")
        gaps = contract_gaps(spec)
        if gaps:
            raise ValueError(f"{phase_name}: contract incomplete, cannot replace (missing {gaps})")
    if fn is None:
        _REPLACED.pop(phase_name, None)
    else:
        _REPLACED[phase_name] = fn


def replaced() -> tuple[str, ...]:
    return tuple(sorted(_REPLACED))


# --------------------------------------------------------------------------- #
# Validation and diagnostics
# --------------------------------------------------------------------------- #
# A replaceable phase must declare at least these (decided by "this phase does this kind of math")
_REQUIRED = ("determinism", "rounding_points")


def contract_gaps(spec: PhaseSpec) -> list[str]:
    """List the field names missing from this phase's contract (used to gatekeep before L3 registration)."""
    gaps = [f for f in _REQUIRED if getattr(spec.numerics, f) in (None, ())]
    if not spec.reads and not spec.writes:
        gaps.append("reads/writes")
    return gaps


def validate() -> list[str]:
    """Return the list of problems (empty = all green).

    This does **not** check whether the numerics are correct (that is guaranteed by measured
    regression); it checks three things at the contract level: (1) the positional parameter
    order matches the pybind signature; (2) every phase's reads/writes are in the buffer table;
    (3) which phases still have incomplete contracts (to be filled before L3 -- this is a to-do
    list, not an error).
    """
    problems: list[str] = []
    for kernel, io in KERNEL_IO.items():
        try:
            ext = _ext(io.ext)
        except Exception as exc:  # noqa: BLE001 - build failures must be reported too
            problems.append(f"{kernel}: extension {io.ext} unavailable ({exc})")
            continue
        doc = getattr(getattr(ext, io.ext_fn), "__doc__", "") or ""
        m = re.match(rf"{re.escape(io.ext_fn)}\((.*)\)\s*->", doc, re.S)
        if not m:
            problems.append(f"{kernel}: cannot read the signature of {io.ext_fn}")
            continue
        declared = [a.split(":")[0].strip() for a in m.group(1).split(",")]
        want = [n for _k, n in io.args]
        if declared != want:
            problems.append(
                f"{kernel}: parameter order disagrees with the pybind signature\n"
                f"      declared {want}\n      actual {declared}"
            )
    for spec in PHASES.values():
        gaps = contract_gaps(spec)
        if gaps:
            problems.append(f"{spec.name}: contract to be filled in {gaps}")
    return problems


def phase_plan(kernels: tuple[str, ...] | None = None) -> str:
    """Print the phase table (these are the phases L1 runs; empty `contract` column = not yet replaceable)."""
    lines = []
    for kernel, io in KERNEL_IO.items():
        if kernels and kernel not in kernels:
            continue
        coop = "coop+split" if io.coop else "plain launch"
        lines.append(
            f"[{kernel}]  {io.ext_fn}  {len(io.phases)} phases  {coop}  grid={io.grid_policy} min={io.min_grid}"
        )
        for spec in phases_of(kernel):
            gaps = contract_gaps(spec)
            mark = "REPL" if spec.name in _REPLACED else ("    " if not gaps else "TODO")
            lines.append(f"  P{spec.index}  {mark}  {spec.does}")
            lines.append(f"       reads {list(spec.reads)}")
            lines.append(f"       writes {list(spec.writes)}")
            if gaps:
                lines.append(f"       contract to be filled in {gaps}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    issues = validate()
    if issues:
        print("Validation problems (contract level, not numerics):")
        for p in issues:
            print("  -", p)
        print()
    if "--plan" in argv:
        print(phase_plan())
    return 0


# Populate the registry (at end of file: when `phase_table` reverse-imports this module, the names above are already defined)
from . import phase_table  # noqa: E402,F401

if __name__ == "__main__":
    # `-m` executes this module a second time as `__main__`, while `phase_table` registers into
    # the phases module **inside the package** -- the two are not the same object, so `KERNEL_IO`
    # looks empty. So go through an in-package reference to guarantee the CLI sees the same registry.
    from . import phases as _canonical

    raise SystemExit(_canonical.main())
