"""Per-phase bit comparison: the acceptance tool for L3 registration (or L2 geometry changes).

**The criterion is not rel-RMS but bit patterns**: run the model once to record each phase's
**entry state**, then replay on **the same input** and compare whether every buffer in the
phase's `writes` matches bit for bit.

```python
from tybok.policies.fastwam.kernels import phases, phase_check

rec = phase_check.Recorder(kernels=["tmt5.ffn"])      # record only the two tmt5.ffn phases
with rec:                                            # recording mode: run normally once
    ...run one layer / one inference (default implementation)...

print(phase_check.report(rec))                       # default implementation vs recording -> self-consistency
phases.replace("tmt5.ffn/P3", host_impl)               # install the implementation to be checked
print(phase_check.report(rec))                       # host operator vs recording -> L3 criterion
```

Recording at the `dispatch()` layer treats all kernels uniformly -- each phase's full set of
named tensors is already there (the `KernelIO` buffer table), so there is no need to write a
separate adapter per kernel, nor to rebuild weights/inputs.

**Aliases must be preserved**: some phases write in place (e.g. in `adit.attn_cross` C_PHASE_6,
`x_in` and `out` are the same memory, and likewise for `vdit.attn_cross`). If replay gives each
of the two names its own fresh allocation, the in-place semantics are lost and the comparison
is invalid. So group by storage: one group shares one fresh copy.

**The entry state must be fully cloned** (keep only aliases, keep no references): the caller
itself rewrites these buffers in place -- video's `modsum` is recomputed every denoise step, the
ping-pong output is overwritten every layer, and the KV cache is rebuilt every chunk. Keeping
references to read-only buffers would make the "recorded input" obtained at replay already be
the content of some later layer/step. The cost is a few hundred MB per recording, traded for a
**self-contained case** (can be saved to disk and replayed across processes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

from . import phases as P

# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    """One call of one phase: entry state + snapshot of the exit at recording time."""

    # fmt: off
    kernel: str
    index: int
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)   # entry
    scalars: dict[str, Any] = field(default_factory=dict)
    exits: dict[str, torch.Tensor] = field(default_factory=dict)     # exit (snapshot of writes)
    # fmt: on

    @property
    def name(self) -> str:
        return f"{self.kernel}/P{self.index}"

    @property
    def spec(self) -> P.PhaseSpec:
        return P.PHASES[self.name]

    def fresh_tensors(self) -> dict[str, torch.Tensor]:
        """Entry tensors for replay: rebuild one copy grouped by storage (aliases and view relationships preserved)."""
        return _snapshot(self.tensors)

    def replay(self, *, impl=None, stream: int | None = None) -> dict[str, torch.Tensor]:
        """Run this phase on the recorded entry, returning a **clone** of each buffer in `writes`.

        `impl=None` uses the registry's currently effective implementation (default, or one replaced via `replace()`).
        """
        tensors = self.fresh_tensors()
        stream = torch.cuda.current_stream().cuda_stream if stream is None else stream
        P.dispatch(self.kernel, self.index, tensors=tensors, scalars=dict(self.scalars), stream=stream)
        torch.cuda.synchronize()
        return {n: tensors[n].detach().clone() for n in self.spec.writes if n in tensors}


def _mutable_buffers(kernel: str) -> set[str]:
    """Buffers in this kernel that any phase rewrites."""
    out: set[str] = set()
    for spec in P.phases_of(kernel):
        out.update(spec.writes)
    return out


def _snapshot(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Snapshot grouped by **storage**: copy the whole storage once, then rebuild each name with
    its original offset/shape/stride.

    WARNING: you cannot `clone()` tensor by tensor: different slices of the same storage would
    each get their **own** copy of the data, losing aliases and view relationships -- e.g. the
    shift/scale/gate of `modsum[i].chunk(6)` are three offset views of one memory block, and
    cloning per tensor makes all three the contents of the first slice, producing inf/nan on replay.
    """
    groups: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for name, t in tensors.items():
        groups.setdefault(t.untyped_storage().data_ptr(), []).append((name, t))
    out: dict[str, torch.Tensor] = {}
    for members in groups.values():
        rep = members[0][1]
        whole = torch.empty(0, dtype=torch.uint8, device=rep.device).set_(rep.untyped_storage())
        dst = whole.clone()
        for name, t in members:
            nt = torch.empty(0, dtype=t.dtype, device=t.device)
            nt.set_(dst.untyped_storage(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))
            out[name] = nt
    return out


class Recorder:
    """Record the entry/exit state each time a phase is called (hooks into `dispatch()`)."""

    def __init__(self, kernels: Iterable[str] | None = None, only: Iterable[str] | None = None, per_phase: int = 1):
        self.kernels = set(kernels) if kernels else None
        self.only = set(only) if only else None
        self.per_phase = per_phase
        self.cases: dict[str, list[Case]] = {}
        self._pending: Case | None = None
        self._prev: Recorder | None = None

    # -- Start / stop -------------------------------------------------------- #
    def __enter__(self) -> "Recorder":
        self._prev = P._RECORDER
        P._RECORDER = self
        return self

    def __exit__(self, *exc) -> None:
        P._RECORDER = self._prev
        P._prev = None

    # -- Hooks called by dispatch -------------------------------------------- #
    def _want(self, kernel: str, index: int) -> bool:
        if index < 0 or (self.kernels and kernel not in self.kernels):
            return False
        name = f"{kernel}/P{index}"
        if self.only and name not in self.only:
            return False
        return len(self.cases.get(name, ())) < self.per_phase

    def enter(self, kernel: str, index: int, tensors: Mapping[str, Any], scalars: Mapping[str, Any]) -> None:
        if not self._want(kernel, index):
            return
        # WARNING: neither can be omitted: (1) the snapshot must be taken **now** -- this phase
        #    is about to modify this memory, so snapshotting at exit would capture the exit state;
        #    (2) even "read-only" ones must be snapshotted -- the caller itself rewrites them in
        #    place (modsum recomputed each step, ping-pong overwritten each layer).
        want = {n: t for n in P.KERNEL_IO[kernel].tensor_names if (t := tensors.get(n)) is not None}
        self._pending = Case(kernel=kernel, index=index, tensors=_snapshot(want), scalars=dict(scalars))

    def exit(self, kernel: str, index: int, tensors: Mapping[str, Any]) -> None:
        case = self._pending
        self._pending = None
        if case is None or case.kernel != kernel or case.index != index:
            return
        case.exits = {n: tensors[n].detach().clone() for n in case.spec.writes if n in tensors}
        self.cases.setdefault(case.name, []).append(case)

    # -- Queries ------------------------------------------------------------- #
    def all_cases(self) -> list[Case]:
        return [c for v in self.cases.values() for c in v]


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass
class BufferDiff:
    name: str
    dtype: str
    numel: int
    identical: bool
    n_diff: int
    frac_diff: float
    max_ulp: int
    max_abs: float
    rel_rms: float


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).to(torch.int32)
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    if t.dtype in (torch.float16,):
        return t.contiguous().view(torch.int16).to(torch.int32)
    return t.contiguous().view(torch.uint8).to(torch.int32)  # fp8 / u8 / u32


def diff_buffer(name: str, got: torch.Tensor, ref: torch.Tensor) -> BufferDiff:
    g, r = got.reshape(-1), ref.reshape(-1)
    gb, rb = _bits(g), _bits(r)
    ne = gb != rb
    n_diff = int(ne.sum())
    gf, rf = g.float(), r.float()
    return BufferDiff(
        name=name,
        dtype=str(ref.dtype),
        numel=r.numel(),
        identical=n_diff == 0,
        n_diff=n_diff,
        frac_diff=n_diff / max(1, r.numel()),
        max_ulp=int((gb - rb).abs().max()) if n_diff else 0,
        max_abs=float((gf - rf).abs().max()) if n_diff else 0.0,
        rel_rms=float((gf - rf).pow(2).mean().sqrt() / rf.pow(2).mean().sqrt().clamp_min(1e-30)) if n_diff else 0.0,
    )


@dataclass
class Verdict:
    case: Case
    buffers: list[BufferDiff]

    @property
    def ok(self) -> bool:
        return all(b.identical for b in self.buffers)


def check(case: Case, *, impl=None) -> Verdict:
    """Replay one case and compare bit patterns buffer by buffer against the phase's **recorded exit**."""
    got = case.replay(impl=impl)
    diffs = []
    for name, ref in case.exits.items():
        if name not in got:
            continue
        diffs.append(diff_buffer(name, got[name], ref))
    return Verdict(case=case, buffers=diffs)


def report(rec: Recorder, *, verbose: bool = True) -> str:
    """Compare all recorded cases and return a summary. Replaced phases are marked."""
    replaced = set(P.replaced())
    lines: list[str] = []
    n_ok = n_bad = 0
    for name in sorted(rec.cases):
        spec = P.PHASES[name]
        tag = "REPL" if name in replaced else "    "
        for i, case in enumerate(rec.cases[name]):
            v = check(case)
            n_ok += v.ok
            n_bad += not v.ok
            head = (
                f"  [{tag}] {name} #{i}  acceptance tier={spec.acceptance}  "
                f"{'bitwise identical OK' if v.ok else '**DIFFERS**'}"
            )
            lines.append(head)
            for b in v.buffers:
                if b.identical:
                    if verbose:
                        lines.append(f"        {b.name:<16} bitwise identical ({b.numel} elements)")
                    continue
                lines.append(
                    f"        {b.name:<16} diff {b.n_diff}/{b.numel} "
                    f"({100 * b.frac_diff:.4f}%)  max {b.max_ulp} ulp  "
                    f"max|diff| {b.max_abs:.3e}  rel-RMS {b.rel_rms:.3e}"
                )
    lines.append(
        f"  -> {n_ok} cases bitwise identical, {n_bad} differ"
        + (f" (replaced phases: {sorted(replaced)})" if replaced else "")
    )
    return "\n".join(lines)
