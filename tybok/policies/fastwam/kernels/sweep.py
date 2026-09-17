"""L2 geometry sweep: compile candidate geometries, measure them on **real tensors**, write the winner back to the macro.

The geometry of a fused kernel is **compile-time** (shapes are `constexpr`, the GEMM's 29 tiles are template arguments), so
"changing geometry" = editing the `FWAM_<kernel>_<phase>_TILES` macro + recompiling. This script walks that whole path, and **does not touch the production source**:

   1) compile  copy `kernels/` in full to scratch, change only the text of the target macro, name the extension by **source-content hash**
          (the same geometry is compiled only once; if the macro was not actually changed the script fails on the spot -- otherwise every number it sweeps would be fake)
   2) measure  measure on **real tensors**: cases are recorded from one real inference with `phase_check.Recorder`, then for each candidate run
          a **bitwise comparison** (against the exit state recorded at capture time) + **interleaved A/B** (CUDA graph timing)
   3) report   ranking + commands that can be pasted back directly; `apply` writes back to the source, then run `geometry` to reconcile

Criteria (two, both required):

* **Bit pattern** -- compare the bit pattern buffer by buffer against the exit state recorded at capture time. Changing **blocking** parameters like BN/BK/STAGES/WP
  should be numerically **bitwise unchanged** (the accumulation order of k does not change); a reported difference has only two possible causes: the axis inherently
  changes the numerics (`SPLIT`/`FA`/`F16P`/`FAP`/`PERSIST`/`PD`/`EPI`/`RESID`), or the computation is wrong. Neither should be adopted **silently** -- so by default only
  bitwise-identical candidates are recommended, and changing the numerics requires an explicit `--allow-drift`.
* **Finiteness** -- any inf/nan in the output is immediately invalid: `FA=1` (intra-level fp16 accumulation) runs fine on small activation values,
  but turns into a whole slab of NaN on production-scale activations.

Two things that "make the sweep pointless if not done":

* **One subprocess per candidate**. A bad geometry does not fail gently -- after crossing the 48KB smem budget it keeps running and writes out of bounds, reporting
  illegal memory access, and **the entire CUDA context is already dead**, so every remaining candidate in the same process is wasted.
  On a crash it marks a "crash" row and moves on to the next candidate.
* **Take the min, not the median**. Timing jitter is larger than the difference being sought; each implementation records its own CUDA graph and they are
  replayed in turn, taking the min, which is the only way to suppress the noise (self-check with the "baseline vs baseline" row: it should hover around 1.000).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_KERNEL_DIR = Path(__file__).resolve().parent
_SCRATCH_ROOT = Path(os.environ.get("FASTWAM_SWEEP_DIR", str(Path.home() / ".cache" / "fastwam_geom_sweep")))
_SRC_SUFFIX = (".cu", ".cuh", ".h", ".cpp")  # full copy: a changed header must also invalidate and recompile

# `fwam_tiles("<label>", FWAM_TILES(<macro>))` -- the **only** self-report form (defined by `geom_report.h`)
_FWAM_TILES_CALL = re.compile(r'fwam_tiles\(\s*"([A-Za-z0-9_]+)"\s*,\s*FWAM_TILES\(\s*(FWAM_[A-Za-z0-9_]+)\s*\)\s*\)')
_GEOM_FN = re.compile(r"const\s+char\s*\*\s*(\w+_geom)\s*\(\s*\)")
_CORE_INC = re.compile(r'#include\s+"(\w+_gemm_core\.cu)"')


# --------------------------------------------------------------------------- #
# read geometry from the source (parse out macro names and axis names, do not copy by hand)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Slot:
    """A sweepable geometry: which phase of which kernel, which macro of which `.cu`, and what the axes are called."""

    # fmt: off
    kernel: str                 # "vdit.ffn"
    label: str                  # "F_PHASE_4" (= key of the geometry table = label of fwam_tiles)
    macro: str                  # "FWAM_VDIT_FFN_F_PHASE_4_TILES"
    file: str                   # "vdit_ffn.cu"
    axes: tuple[str, ...]       # template parameter names, in order (= positional order of --tiles)
    # fmt: on


def _strip_line_comments(text: str) -> str:
    """Replace `// ...` with same-length spaces (keeps indices aligned; parsing template arguments must skip the `<` `>`, and `,` inside comments)."""
    out = []
    for line in text.split("\n"):
        i = line.find("//")
        out.append(line if i < 0 else line[:i] + " " * (len(line) - i))
    return "\n".join(out)


def _template_axes(text: str) -> tuple[str, ...]:
    """Template parameter names of `fwam_fp8_gemm_body` (in order) = positional order of `--tiles`."""
    code = _strip_line_comments(text)
    i = code.index("void fwam_fp8_gemm_body")
    j = code.rindex("template", 0, i)
    k = code.index("<", j) + 1
    depth, cur, parts = 1, [], []
    while depth:
        c = code[k]
        if c == "<":
            depth += 1
            cur.append(c)
        elif c == ">":
            depth -= 1
            if depth == 0:
                parts.append("".join(cur))
                break
            cur.append(c)
        elif c == "," and depth == 1:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        k += 1
    names = []
    for p in parts:
        m = re.search(r"\b([A-Za-z_]\w*)\s*(?:=[^,]*)?$", p.strip())
        if m is None:
            raise SystemExit(f"cannot parse template parameter name: {p!r}")
        names.append(m.group(1))
    if len(names) != 29:
        raise SystemExit(f"parsed {len(names)} template parameters (expected 29): {names}")
    return tuple(names)


def slots() -> dict[str, Slot]:
    """Scan `kernels/*.cu` and list all sweepable geometries (key = "<kernel>/<label>")."""
    from . import geometry as G

    fn2kernel = {rep.csym: k for k, rep in G._REPORTERS.items()}
    axes_cache: dict[str, tuple[str, ...]] = {}
    out: dict[str, Slot] = {}
    for path in sorted(_KERNEL_DIR.glob("*.cu")):
        text = path.read_text()
        if "fwam_tiles(" not in text:
            continue
        for m in _FWAM_TILES_CALL.finditer(text):
            # which *_geom() this self-report is in -- search backwards for the nearest function definition
            fns = list(_GEOM_FN.finditer(text, 0, m.start()))
            if not fns:
                continue
            kernel = fn2kernel.get(fns[-1].group(1))
            if kernel is None:
                continue
            if kernel not in axes_cache:
                inc = _CORE_INC.search(text)
                if inc is None:
                    raise SystemExit(f"{path.name}: cannot find the include of the GEMM body")
                axes_cache[kernel] = _template_axes((_KERNEL_DIR / inc.group(1)).read_text())
            out[f"{kernel}/{m.group(1)}"] = Slot(kernel, m.group(1), m.group(2), path.name, axes_cache[kernel])
    return out


def slot_for(smap: dict[str, Slot], phase_name: str, geom: str | None = None) -> Slot:
    """`--phase vdit.ffn/P4` (+ optional `--geom`, only when the phase declares multiple tile labels) -> Slot."""
    from . import phases as P

    spec = P.PHASES.get(phase_name)
    if spec is None:
        raise SystemExit(f"unregistered phase {phase_name} (available: {', '.join(sorted(P.PHASES))})")
    if not spec.tiles:
        raise SystemExit(
            f"{phase_name}: this phase has no sweepable geometry ({spec.does}) -- only phases that go "
            f"through the shared GEMM body carry a `tiles` label. For hand-written kernels (adit.*) the geometry is directly `constexpr`; edit the source."
        )
    if geom is None:
        if len(spec.tiles) > 1:
            raise SystemExit(f"{phase_name} has multiple geometries {list(spec.tiles)}; pick one with --geom")
        label = spec.tiles[0]
    else:
        if geom not in spec.tiles:
            raise SystemExit(f"{phase_name}'s geometries are {list(spec.tiles)}, none is {geom!r}")
        label = geom
    name = f"{spec.kernel}/{label}"
    if name not in smap:  # the phase table says this label exists, but the source does not -> the two are out of sync
        raise SystemExit(
            f"{name}: the phase table declares label {label!r}, but the source has no "
            f"corresponding fwam_tiles self-report -- phase table and kernel are out of sync"
        )
    return smap[name]


# --------------------------------------------------------------------------- #
# patch the macro + compile a variant (**does not touch the production source**)
# --------------------------------------------------------------------------- #
def _read_define_body(text: str, m: re.Match) -> tuple[str, int]:
    """**Logical line** of the macro (`\\` continuations count as the same line): returns (value text, end position = where the last line's newline is)."""
    i = j = m.end()
    while True:
        nl = text.index("\n", j)
        line = text[j:nl]
        j = nl + 1
        if not line.rstrip().endswith("\\"):
            break
    return text[i:j], j


def _define_re(macro: str) -> re.Pattern:
    return re.compile(rf"^#define\s+{macro}\b", re.M)


def current_values(slot: Slot) -> tuple[int, ...]:
    """The current value group of this macro in the production source."""
    text = (_KERNEL_DIR / slot.file).read_text()
    m = _define_re(slot.macro).search(text)
    if m is None:
        raise SystemExit(f"no macro {slot.macro} in {slot.file}")
    body, _ = _read_define_body(text, m)
    return tuple(int(v) for v in re.findall(r"-?\d+", body))


def patch_macro(text: str, macro: str, values: Sequence[int]) -> str:
    """Change values **in place**: leave commas, spaces, and `\\` continuation positions untouched, replacing only the digits.

    Do not generate a new `#define`: macros wrap lines inconsistently, and rewriting would make `apply`'s diff unrecognizable.
    In-place editing has another advantage: when the value has not changed it is **literally identical**, so a "fake sweep" is caught on the spot.
    """
    m = _define_re(macro).search(text)
    if m is None:
        raise SystemExit(f"no macro {macro} in the source")
    body, end = _read_define_body(text, m)
    residue = re.sub(r"-?\d+", "", body)
    if residue.strip(" \t\r\n,\\"):
        raise SystemExit(
            f"{macro}: the macro body has non-numeric content {residue.strip()!r} -- "
            f"cannot edit in place; handle it by hand"
        )
    nums = list(re.finditer(r"-?\d+", body))
    if len(nums) != len(values):
        raise SystemExit(
            f"{macro}: the macro has {len(nums)} values but {len(values)} were given -- "
            f"count mismatch (adding/removing an axis is a structural change; write the macro by hand)"
        )
    out = body
    for mm, v in zip(reversed(nums), reversed(values)):
        out = out[: mm.start()] + str(v) + out[mm.end() :]
    return text[: m.end()] + out + text[end:]


def _family() -> dict[str, str]:
    """All source under `kernels/` (a full copy => the variant differs from production by only that one macro)."""
    return {p.name: p.read_text() for p in sorted(_KERNEL_DIR.iterdir()) if p.suffix in _SRC_SUFFIX}


def _hash(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name, text in sorted(files.items()):
        h.update(f"{name}\0{text}\0".encode())
    return h.hexdigest()[:16]


@dataclass
class Variant:
    slot: Slot
    values: tuple[int, ...]
    key: str  # source-content hash (= extension-name suffix)
    src_dir: Path


def materialize(slot: Slot, values: Sequence[int]) -> Variant:
    """Drop the macro-patched source into scratch (named by content hash, atomic on disk, reusable)."""
    files = _family()
    original = files[slot.file]
    files[slot.file] = patch_macro(original, slot.macro, values)
    if files[slot.file] == original:
        raise SystemExit(
            f"{slot.macro}: after patching it is literally identical to the original -- did the values not change?"
        )
    key = _hash(files)
    d = _SCRATCH_ROOT / key
    if not (d / ".ok").exists():
        tmp = _SCRATCH_ROOT / f"{key}.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name, text in files.items():
            (tmp / name).write_text(text)
        (tmp / ".ok").write_text("")
        try:
            tmp.rename(d)  # atomic: a crash does not leave a half-written source tree
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)  # someone else landed it first
    return Variant(slot, tuple(values), key, d)


def _ext_of(kernel: str) -> str:
    from . import phases as P

    return P.KERNEL_IO[kernel].ext


def ext_name(ext_key: str, key: str) -> str:
    return f"fw_geom_{ext_key}_{key}"


def make_ext(ext_key: str, name: str, src_dir: str) -> Any:
    from . import SOURCES, build_variant

    return build_variant(name, src_dir, SOURCES[ext_key])


def build(variant: Variant) -> Any:
    ext_key = _ext_of(variant.slot.kernel)
    return make_ext(ext_key, ext_name(ext_key, variant.key), str(variant.src_dir))


@contextlib.contextmanager
def _using_ext(ext_key: str, ext: Any):
    """Make `phases.dispatch` go through this variant extension (the default implementation looks up the cache by ext key)."""
    from . import phases as P

    missing = object()
    prev = P._EXT_CACHE.get(ext_key, missing)  # noqa: SLF001
    P._EXT_CACHE[ext_key] = ext  # noqa: SLF001
    try:
        yield
    finally:
        if prev is missing:
            P._EXT_CACHE.pop(ext_key, None)  # noqa: SLF001
        else:
            P._EXT_CACHE[ext_key] = prev  # noqa: SLF001


# --------------------------------------------------------------------------- #
# capture: run one real inference and record each phase's entry state
# --------------------------------------------------------------------------- #
def capture(
    checkpoint: str,
    *,
    out: str,
    kernels: Iterable[str] | None = None,
    per_phase: int = 1,
    model_type: str | None = None,
    device: str = "cuda",
    cameras: int | None = None,
    steps: int | None = None,
    sampler: str = "euler",
    seed: int = 0,
    split: bool = True,
    text_encoder_device: str = "cuda",
) -> None:
    """Run one inference (default **split-phase** form -- the cooperative form is a single launch with no phases) and dump a case for each phase.

    `text_encoder_device="cuda"` differs from the worker default (`cpu`): with UMT5 left on CPU the text-fusion path is blocked by the engine ("UMT5 is not fp8-resident"),
    so **not a single one of tmt5's 9 phases can be recorded**. One capture must cover all 41 phases, so the default here moves it to the GPU (the embedding table stays on CPU to save memory).
    """
    import torch

    from tybok.registry import create_engine

    from . import phase_check

    engine = create_engine(
        checkpoint,
        model_type=model_type,
        device=device,
        video_fp8=True,
        action_fp8=True,
        text_fp8=True,
        pack_qkv=True,
        num_steps=steps,
        sampler=sampler,
        seed=seed,
        text_encoder_device=text_encoder_device,
        text_emb_cpu=True,
        action_fused=True,
        video_fused=True,
        text_fused=True,
        action_fused_split=split,
        video_fused_split=split,
        text_fused_split=split,
    )
    # keep the frame assembly identical to the engine's own `_warmup`: `describe()["cameras"]` returns
    # **full names** (`observation.images.image`), while `make_frame` adds the prefix itself -> without
    # stripping you get a double prefix and the preprocessor directly reports "missing every image feature".
    spec = engine.describe()
    h, w = spec["resize"][1], spec["resize"][0]
    cams = spec["cameras"] if cameras is None else spec["cameras"][:cameras]
    g = torch.Generator().manual_seed(seed)
    images = {cam.split("observation.images.")[-1]: torch.rand(3, h, w, generator=g) for cam in cams}
    n_state = getattr(engine.config, "proprio_dim", None) or 8
    frame = engine.make_frame(images, torch.rand(n_state, generator=g), "pick up the cup")

    rec = phase_check.Recorder(kernels=kernels, per_phase=per_phase)
    with rec:  # recording mode: the default implementation just runs once
        engine.predict_action_chunk(frame)
    n = len(rec.all_cases())
    if n == 0:
        raise SystemExit(
            "no case was recorded at all -- did the engine not take the split-phase form? "
            "(the cooperative form is a single launch and has no phases)"
        )
    torch.cuda.synchronize()
    torch.save({"cases": rec.cases, "checkpoint": checkpoint}, out)
    print(f"recorded {n} cases ({len(rec.cases)} phases) -> {out}")


def _load_cases(path: str) -> dict[str, list]:
    import torch

    try:
        blob = torch.load(path, weights_only=False)
    except TypeError:  # old torch has no weights_only
        blob = torch.load(path)
    return blob["cases"] if isinstance(blob, dict) and "cases" in blob else blob


# --------------------------------------------------------------------------- #
# measure: bitwise comparison + interleaved A/B
# --------------------------------------------------------------------------- #
_WARMUP = 5  # warmup iterations before capture (to settle the clock/caches)


def _capture(fn, iters: int):
    """Record `iters` dispatches into a CUDA graph.

    Only the dispatch is recorded in the graph: visibility between phases is determined by stream order,
    and single-phase replay is idempotent, so running the same tensors in place repeatedly does not
    affect the timing.
    """
    import torch

    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return g


def _replay_time(g, iters: int, rounds: int) -> list[float]:
    """Replay several times, timing each one separately. **Timing each replay** (instead of one big block) is so that the min can be taken later."""
    import torch

    out = []
    g.replay()  # discard the first (the first replay has extra overhead)
    torch.cuda.synchronize()
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) / iters * 1e3)
    return out


def _runner(case):
    """Closure used for timing: **takes the current stream on every call**, not the one from recording time.

    WARNING: this is mandatory: during CUDA graph capture the "current stream" is the capture stream, while the one recorded is the default
    stream (0). Kernels launched onto the legacy default stream **do not enter the graph** (and it is not a capture error), so the graph is empty,
    replay does nothing, and the reported "elapsed time" looks like it got a thousand times faster.
    """
    import torch

    from . import phases as P

    tensors = case.tensors
    base = dict(case.scalars)
    kernel, index = case.kernel, case.index

    def run() -> None:
        s = torch.cuda.current_stream().cuda_stream
        base["stream"] = s
        P.dispatch(kernel, index, tensors=tensors, scalars=base, stream=s)

    return run


@dataclass
class Row:
    """One result row for a candidate. A non-empty `note` = the candidate did not run (failed to compile / crashed / timed out)."""

    # fmt: off
    values: tuple[int, ...]
    finite: bool = True
    identical: bool = True
    t_med: float = float("nan")
    ratio: float = float("nan")
    diff: str = "bitwise identical"
    note: str = ""                  # failure reason (non-empty = invalid candidate)
    prod_ok: bool = True            # within the same subprocess the baseline is also bitwise identical to the recorded exit (self-check)
    # fmt: on

    @property
    def bitwise(self) -> bool:
        return self.finite and self.identical

    @property
    def ok(self) -> bool:
        return not self.note


def _replay(case, ext_key: str, ext: Any):
    """Replay this case on the given extension; returns (exit tensors, per-buffer differences vs the recorded exit)."""
    from . import phase_check

    with _using_ext(ext_key, ext):
        got = case.replay()
    return got, [phase_check.diff_buffer(n, t, case.exits[n]) for n, t in got.items() if n in case.exits]


def _diff_summary(diffs: list) -> str:
    bad = [d for d in diffs if not d.identical]
    if not bad:
        return "bitwise identical"
    return " + ".join(f"{d.name} {d.n_diff}/{d.numel} ({100 * d.frac_diff:.3f}%)" for d in bad)


# ---- subprocess: one per candidate ---------------------------------------------------------
# A bad geometry is **not** a gentle failure: for vdit.ffn's down phase raising BN from 48 to 64 pushes the smem requirement past the 48KB budget and the kernel keeps running,
# writing out of bounds -- reporting "illegal memory access", while **the entire CUDA context is already dead**, wasting every remaining candidate in the process.
# So start one subprocess per candidate: on a crash mark a "crash" row and move on to the next.
def _one(
    cases_path: str,
    phase_name: str,
    *,
    geom: str | None,
    values: list[int],
    reps: int,
    iters: int,
    src_dir: str | None = None,
) -> int:
    """Subprocess entry point: measure one candidate and print the result as one line of JSON to stdout."""
    import json

    from . import phases as P

    slot = slot_for(slots(), phase_name, geom)
    cases = _load_cases(cases_path).get(phase_name)
    if not cases:
        raise SystemExit(f"no {phase_name} in the cases")
    case = cases[0]
    ext_key = _ext_of(slot.kernel)
    prod = P._ext(ext_key)  # noqa: SLF001
    if src_dir is None:  # baseline row: prod vs prod (which also self-checks the timing method)
        ext = prod
        got, diffs = _replay(case, ext_key, ext)
        prod_ok = all(d.identical for d in diffs)
    else:
        ext = make_ext(ext_key, ext_name(ext_key, Path(src_dir).name), src_dir)
        got_prod, pdiffs = _replay(case, ext_key, prod)  # re-measure the baseline in this process (paired A/B)
        got, diffs = _replay(case, ext_key, ext)
        prod_ok = all(d.identical for d in pdiffs)
        del got_prod
    finite = all(bool(t.isfinite().all()) for t in got.values())
    tp, tv = _ab_time(case, ext_key, prod, ext, iters, reps)
    print(
        "RESULT "
        + json.dumps(
            {
                "t_prod": tp,
                "t_var": tv,
                "ratio": tv / tp if tp else 0.0,
                "identical": all(d.identical for d in diffs),
                "finite": finite,
                "diff": _diff_summary(diffs),
                "prod_ok": prod_ok,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _ab_time(case, ext_key: str, prod: Any, ext: Any, iters: int, rounds: int) -> tuple[float, float]:
    """Interleaved A/B timing: each implementation records its own graph, then they are **replayed in turn**, taking the min of each.

    Two traps (either makes the reported numbers completely untrustworthy):

    * **Do not use the median**. Per-round jitter is larger than the difference being sought; the median just moves the noise around. Taking the **min**
      is the robust estimator in microbenchmarks (the min corresponds to "the round that was not disturbed").
    * **Do not record graphs inside the measurement loop**. Capture takes tens of milliseconds and perturbs state; record one graph per implementation, then only replay.
    """
    runner = _runner(case)
    graphs = {}
    for side, impl in (("prod", prod), ("var", ext)):
        with _using_ext(ext_key, impl):
            graphs[side] = _capture(runner, iters)
    ts = {"prod": [], "var": []}
    for r in range(rounds * 2):  # replay in turn: swap the order on odd/even rounds
        for side in ("prod", "var") if r % 2 == 0 else ("var", "prod"):
            ts[side] += _replay_time(graphs[side], iters, 1)
    return min(ts["prod"]), min(ts["var"])


def run(
    cases_path: str,
    phase_name: str,
    *,
    geom: str | None = None,
    full: Iterable[str] = (),
    axes_raw: str = "",
    reps: int = 5,
    iters: int = 50,
    allow_drift: bool = False,
    timeout: float = 600.0,
    allow_structural: bool = False,
) -> int:
    import subprocess

    slot = slot_for(slots(), phase_name, geom)
    cases = _load_cases(cases_path).get(phase_name)
    if not cases:
        raise SystemExit(f"no {phase_name} in the cases (was this phase not reached during capture?)")
    base = current_values(slot)
    cands = _candidates(full, axes_raw, slot, base)
    for values in cands:
        _check_sweepable(slot, base, values, allow_structural)
    prod_hash = _hash(_family())
    budget = _smem_budget(slot.kernel)

    print(f"{phase_name}  geometry {slot.label}  macro {slot.macro} ({slot.file})")
    for name, v in zip(slot.axes, base):
        print(f"    {name:<7} = {v}")
    print(
        f"{len(cases)} cases ({cases_path}); {len(cands)} candidates; "
        f"{reps} interleaved A/B rounds x {iters} iters each; one subprocess per candidate (timeout {timeout:.0f}s)\n"
    )

    rows: list[Row] = []
    for values in cands:
        src_dir = None
        if values != base:
            est = _smem_estimate(slot, values)
            if budget and est > budget:
                print(
                    f"  !! {_fmt_vals(slot, base, values)}: estimated smem {est} > budget {budget} B"
                    f" -- most likely an out-of-bounds write (it will crash on this after compiling, just a heads-up)"
                )
            variant = materialize(slot, values)
            if variant.key == prod_hash:
                raise SystemExit(
                    f"{slot.macro}: values changed but the source hash did not -- the macro was "
                    f"not patched, so every number swept would be fake"
                )
            try:  # compile failure (including static_assert) -> mark a row
                build(variant)
            except Exception as exc:  # noqa: BLE001
                rows.append(Row(values, note=f"failed to compile: {_tail(str(exc))}"))
                print(f"  ... {_fmt_vals(slot, base, values)}: failed to compile")
                continue
            src_dir = str(variant.src_dir)
        rows.append(
            _measure(subprocess, cases_path, phase_name, geom, values, reps, iters, src_dir, timeout, slot, base)
        )
    _report(slot, base, rows, allow_drift)
    if any(r.ok and not r.prod_ok for r in rows):
        print(
            "\nWARNING:  in some subprocess **the baseline itself** disagrees with the recorded exit -- recording/replay is not reproducible, "
            "so this round's numbers are untrustworthy (first check whether the phase is deterministic: reduction in the registry should be fixed)"
        )

    usable = [r for r in rows if _acceptable(r, base, allow_drift)]
    if not usable:
        print("\nno usable candidate (all failed to compile / crashed / changed the numerics / non-finite)")
        return 1
    win = min(usable, key=lambda r: r.ratio)  # rank by the **paired ratio**, not by the cross-process absolute value
    print(
        f"\nfastest usable: {_fmt_vals(slot, base, win.values)}  "
        f"{win.t_med:.1f}us ({win.ratio:.3f}x paired baseline {win.t_med / win.ratio:.1f}us)"
    )
    if win.values == base:
        print("the fastest is the current geometry -- nothing better, no need to touch the macro")
    else:
        print("\nwrite back:")
        print("  " + _apply_cmd(phase_name, geom, win.values))
    return 0


def _measure(subprocess, cases_path, phase_name, geom, values, reps, iters, src_dir, timeout, slot, base) -> Row:
    """Run one candidate's subprocess and collect one JSON line into a Row. Returns a Row even on a crash/timeout (with the reason in note)."""
    import json

    cmd = [
        sys.executable,
        "-m",
        f"{__package__}.sweep",
        "_one",
        cases_path,
        "--phase",
        phase_name,
        "--values",
        ",".join(str(v) for v in values),
        "--reps",
        str(reps),
        "--iters",
        str(iters),
    ]
    if geom:
        cmd += ["--geom", geom]
    if src_dir:
        cmd += ["--src-dir", src_dir]
    label = _fmt_vals(slot, base, values)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  ... {label}: timeout ({timeout:.0f}s)")
        return Row(values, note=f"timeout ({timeout:.0f}s)")
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if p.returncode != 0 or line is None:
        why = _tail((p.stderr or p.stdout).strip()) or f"exit code {p.returncode}"
        print(f"  ... {label}: failed ({why})")
        return Row(values, note=why)
    d = json.loads(line[len("RESULT ") :])
    row = Row(
        values,
        finite=d["finite"],
        identical=d["identical"],
        t_med=d["t_var"],
        ratio=d["ratio"],
        diff=d["diff"],
        prod_ok=d["prod_ok"],
    )
    print(f"  ... {label}: {row.diff}  {row.t_med:.1f}us  {row.ratio:.3f}x")
    return row


def _tail(text: str, n: int = 200) -> str:
    """Pick the truly relevant line out of dozens of lines of error output.

    WARNING: do not "find the last line containing error": CUDA's boilerplate tail ("CUDA kernel errors might be asynchronously reported...")
    contains error and would override `illegal memory access`. Prefer the exception head of the traceback (`torch.AcceleratorError: ...`),
    then the compiler's `error:`, and only fall back last.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    for pat in (r"^[\w.]+(Error|Exception)\b", r"\berror\b\s*:", r"\berror\b"):
        hit = next((ln for ln in reversed(lines) if re.search(pat, ln, re.I)), None)
        if hit:
            return hit[-n:]
    return lines[-1][-n:]


# Axes that a macro-only change **cannot sweep**: they change the contract between kernel and host, not the blocking.
# The premise of a sweep is "swap the blocking, and the inputs, outputs, and numerics stay the same" -- these axes violate that premise, and they do **not** fail gently.
_STRUCTURAL = {
    "PERSIST": "changes the grid shape (`T0_ = PERSIST ? blockIdx.x : blockIdx.x + blockIdx.y*MT_`) -- "
               "the persistent form relies on a 1D grid-stride, the non-persistent form needs a 2D "
               "tile grid, and the host's `*_split_grid()` is computed for the persistent form. Changing "
               "only the macro = using a 1D grid as if it were 2D, out-of-bounds writes (measured: whole out NaN)",
    "SPLIT": "changes the partial output path (whether a partial buffer is needed, who does the reduction)",
    "PD": "split mode writes partial directly (drops the smem staging and one barrier)",
    "EPI": "fused epilogue writes gbuf/raw1 (implemented only under SPLIT=1)",
    "RESID": "epilogue appends a residual term (changes the written exit)",
    "APATH": "A goes through LDG->reg->STS (a different issue path and register staging)",
    "FA": "intra-level fp16 accumulation -- **the numerics tier changes**, and production-scale activations overflow to NaN",
    "F16P": "partial uses fp16 (same as above, overflow risk)",
    "FAP": "FA's promotion period (meaningful only when FA is on)",
}  # fmt: skip


def _check_sweepable(slot: Slot, base: tuple[int, ...], values: tuple[int, ...], allow_structural: bool) -> None:
    """Stop axes that "a macro-only change cannot sweep" -- rather than compiling and crashing, explain why."""
    if allow_structural:
        return
    bad = [a for a, b, v in zip(slot.axes, base, values) if b != v and a in _STRUCTURAL]
    if bad:
        detail = "\n".join(f"    {a}: {_STRUCTURAL[a]}" for a in bad)
        raise SystemExit(
            f"these axes cannot be swept by changing only the macro ({slot.macro}):\n{detail}\n"
            f"  to actually try them you must also change the corresponding part of the source "
            f"(grid / partial path / numerics tier), which is no longer a geometry sweep.\n"
            f"  if you really must compile them: --allow-structural (at your own risk)."
        )


def _smem_budget(kernel: str) -> int | None:
    """The dynamic smem this kernel is launched with (the constant in the host launcher, already declared in the phase table)."""
    from . import phases as P

    return P.KERNEL_IO[kernel].geometry.get("SMEM")


def _smem_estimate(slot: Slot, values: tuple[int, ...]) -> int:
    """Estimate smem by the body's formula: `(BM*row)*SA + (BN*row)*SB`, with row stride BK+16 when XS=2 (dense+pad).

    **This is only a warning**: the kernel itself is authoritative -- if the estimate is too high it writes out of bounds (measured: crashes on illegal memory access),
    and if the estimate misses, the subprocess catches it. When unsure about XS, always estimate without pad (too small = only under-reports).
    """
    p = dict(zip(slot.axes, values))
    row = p["BK"] + (16 if p.get("XS") == 2 else 0)  # A/B rings have independent depths
    return p["BM"] * row * p["SA"] + p["BN"] * row * p["SB"]


def _candidates(full: Iterable[str], axes_raw: str, slot: Slot, base: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Candidates = the full `--tiles` tuples + the Cartesian product of `--axes NAME=v1,v2`; the baseline always comes first."""
    out = [tuple(base)]
    for text in full:
        vals = tuple(int(v) for v in re.findall(r"-?\d+", text))
        if len(vals) != len(slot.axes):
            raise SystemExit(
                f"--tiles needs {len(slot.axes)} values ({' '.join(slot.axes)}), but {len(vals)} were given"
            )
        out.append(vals)
    if axes_raw:
        grid: dict[str, list[int]] = {}
        for group in axes_raw.split():
            name, _, vals = group.partition("=")
            if name not in slot.axes:
                raise SystemExit(f"unknown geometry axis {name!r}; the axes of this kernel are {' '.join(slot.axes)}")
            grid[name] = [int(v) for v in vals.split(",") if v.strip()]
        for combo in itertools.product(*grid.values()):
            vals = list(base)
            for name, v in zip(grid, combo):
                vals[slot.axes.index(name)] = v
            out.append(tuple(vals))
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _acceptable(row: Row, base: tuple[int, ...], allow_drift: bool) -> bool:
    """Usable = ran  and  finite  and (is the current geometry / bitwise identical / drift explicitly allowed)."""
    return row.ok and row.finite and (row.values == base or row.bitwise or allow_drift)


def _fmt_vals(slot: Slot, base: tuple[int, ...], values: tuple[int, ...]) -> str:
    changed = [f"{a}={v}" for a, b, v in zip(slot.axes, base, values) if b != v]
    return " ".join(changed) if changed else "= current geometry"


def _apply_cmd(phase_name: str, geom: str | None, values: Sequence[int]) -> str:
    tiles = ", ".join(str(v) for v in values)
    g = f" --geom {geom}" if geom else ""
    return f'python -m tybok.policies.fastwam.kernels.sweep apply --phase {phase_name}{g} --tiles "{tiles}"'


def _report(slot: Slot, base: tuple[int, ...], rows: list[Row], allow_drift: bool) -> None:
    w = max(len(_fmt_vals(slot, base, r.values)) for r in rows)
    # "relative" is measured **paired** within the same subprocess (the baseline is re-measured in the same process); this is the only comparable number;
    # "min us" is absolute and drifts a few % across subprocesses, so it is for reference only.
    print(f"{'candidate':<{w}}  {'bit pattern':<38} {'min us':>8} {'rel':>7}")
    for r in rows:
        label = _fmt_vals(slot, base, r.values)
        if not r.ok:  # failed to compile / crashed / timed out
            tag = f"**invalid: {r.note}**"
            cells = f"{'-':>8} {'-':>7}"
        else:
            if not r.finite:
                tag = "**non-finite (inf/nan) -> invalid**"
            elif not r.identical:
                tag = "numerics changed: " + r.diff
            else:
                tag = "bitwise identical"
            cells = f"{r.t_med:8.1f} {r.ratio:7.3f}"
        mark = "" if _acceptable(r, base, allow_drift) else "   <- not adopted"
        print(f"{label:<{w}}  {tag:<38} {cells}{mark}")


# --------------------------------------------------------------------------- #
# write back
# --------------------------------------------------------------------------- #
def apply(phase_name: str, values: Sequence[int], *, geom: str | None = None, dry_run: bool = False) -> int:
    """Write the selected geometry back into the macro in the production source. Then recompile + `geometry` reconcile (the order must not be reversed)."""
    slot = slot_for(slots(), phase_name, geom)
    path = _KERNEL_DIR / slot.file
    text = path.read_text()
    new = patch_macro(text, slot.macro, values)
    if new == text:
        print(f"{slot.macro} is already this value group, nothing changed")
        return 0
    if dry_run:
        i = new.index(f"#define {slot.macro}")
        print(new[i : i + 240])
        return 0
    path.write_text(new)
    print(f"wrote {slot.file}::{slot.macro}\n")
    print("then do (the order must not be reversed):")
    print(
        "  1) recompile: it is rebuilt automatically on the next import (the source changed); if unsure, delete the build dir"
    )
    print(f"     rm -rf ~/.cache/torch_extensions/*/fastwam_{_ext_of(slot.kernel)}_fused_ext")
    print("  2) reconcile + update the geometry table (it reports which fields are inconsistent):")
    print("     python -m tybok.policies.fastwam.kernels.geometry")
    print("     python -m tybok.policies.fastwam.kernels.geometry --bootstrap")
    print(
        "  3) bitwise regression across the three families: blocking parameters (BN/BK/STAGES/WP) should **not change the numerics**;"
    )
    print(
        "     if the e2e numbers differ from before, the selected axis changes the numerics -- go back to run and look at the bit-pattern column."
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_list(phase: str | None) -> int:
    smap = slots()
    if phase:
        slot = slot_for(smap, phase)
        base = current_values(slot)
        print(f"{phase}  geometry {slot.label}  macro {slot.macro} ({slot.file})")
        for name, v in zip(slot.axes, base):
            print(f"    {name:<7} = {v}")
        print(f"    ({len(slot.axes)} axes, positional order = the order of --tiles)")
        return 0
    for name in sorted(smap):
        s = smap[name]
        print(f"{name:<24} {s.macro:<32} {s.file:<22} {len(s.axes)} axes")
    return 0


def _prune() -> int:
    n = len(list(_SCRATCH_ROOT.iterdir())) if _SCRATCH_ROOT.exists() else 0
    shutil.rmtree(_SCRATCH_ROOT, ignore_errors=True)
    import torch.utils.cpp_extension as cpp

    base = Path(getattr(cpp, "TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache" / "torch_extensions")))
    dirs = list(base.rglob("fw_geom_*"))
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    print(f"deleted {n} scratch source dirs + {len(dirs)} variant extension builds ({base})")
    return 0


def _add_list_args(parser: argparse.ArgumentParser) -> None:
    """``sweep list``: print the sweepable geometries, or one phase's axes and values."""
    parser.add_argument("--phase", default=None)


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    """``sweep capture``: run one real inference and record each phase entry."""
    parser.add_argument("--model", required=True, help="path to the model checkpoint dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kernels", default=None, help="comma-separated; record only these kernels")
    parser.add_argument("--per-phase", type=int, default=1)
    parser.add_argument("--model-type", default=None, help="backend override (default: config.json 'type')")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cameras", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text-encoder-device",
        default="cuda",
        help="defaults to cuda (unlike the worker's cpu): with UMT5 left on CPU "
        "the text-fusion path is blocked and no tmt5 phase can be recorded",
    )
    parser.add_argument(
        "--coop",
        action="store_true",
        help="record the cooperative form (no phases, not sweepable; default: split form)",
    )


def _add_timing_args(parser: argparse.ArgumentParser) -> None:
    """``run`` / ``_one``: the timing knobs both sides of a comparison share."""
    parser.add_argument(
        "--reps", type=int, default=5, help="timing rounds per side (each counts %d runs, takes the min)" % 5
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="dispatches per CUDA graph; larger amortises capture overhead"
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """``sweep run``: sweep a candidate set (bitwise comparison + interleaved A/B)."""
    parser.add_argument("cases")
    parser.add_argument("--phase", required=True, help="e.g. vdit.ffn/P4")
    parser.add_argument("--geom", default=None, help="select among a phase's tile labels, e.g. C_PHASE_2_2G")
    parser.add_argument(
        "--axes", default="", help='e.g. "BN=32,48,64 BK=128" (omitted axes keep their current macro values)'
    )
    parser.add_argument("--tiles", action="append", default=[], help="a full 29-tuple (repeatable)")
    _add_timing_args(parser)
    parser.add_argument("--allow-drift", action="store_true", help="allow candidates whose recommended values differ")
    parser.add_argument(
        "--allow-structural",
        action="store_true",
        help="also compile axes that macros alone cannot sweep (PERSIST/SPLIT/EPI/...; see _STRUCTURAL)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-candidate subprocess timeout in seconds (bad geometries crash or hang)",
    )


def _add_one_args(parser: argparse.ArgumentParser) -> None:
    """``sweep _one``: the internal subprocess that measures one candidate."""
    parser.add_argument("cases")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--geom", default=None)
    parser.add_argument("--values", required=True)
    parser.add_argument(
        "--src-dir", default=None, help="omit to measure the production extension itself (baseline row)"
    )
    _add_timing_args(parser)


def _add_apply_args(parser: argparse.ArgumentParser) -> None:
    """``sweep apply``: write the selected geometry back into the source macros."""
    parser.add_argument("--phase", required=True)
    parser.add_argument("--geom", default=None)
    parser.add_argument("--tiles", required=True)
    parser.add_argument("--dry-run", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    """The sweep CLI: ``list`` / ``capture`` / ``run`` / ``_one`` / ``apply`` / ``prune``."""
    ap = argparse.ArgumentParser(prog="sweep", description="fastwam kernel geometry sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list sweepable geometries; with --phase, print its axes and current values")
    _add_list_args(p)

    p = sub.add_parser("capture", help="run one real inference and record each phase entry")
    _add_capture_args(p)

    p = sub.add_parser("run", help="sweep a candidate set: bitwise comparison + interleaved A/B")
    _add_run_args(p)

    p = sub.add_parser("_one", help="(internal) subprocess: measure one candidate and print one JSON line")
    _add_one_args(p)

    p = sub.add_parser("apply", help="write the selected geometry back into the source macros")
    _add_apply_args(p)

    sub.add_parser("prune", help="delete the sweep scratch and variant extension builds")
    return ap


def _dispatch(args: argparse.Namespace) -> int:
    """Run the subcommand selected on the command line."""
    if args.cmd == "list":
        return _cmd_list(args.phase)
    if args.cmd == "capture":
        capture(
            args.model,
            out=args.out,
            kernels=args.kernels.split(",") if args.kernels else None,
            per_phase=args.per_phase,
            model_type=args.model_type,
            device=args.device,
            cameras=args.cameras,
            steps=args.steps,
            sampler=args.sampler,
            seed=args.seed,
            split=not args.coop,
            text_encoder_device=args.text_encoder_device,
        )
        return 0
    if args.cmd == "run":
        return run(
            args.cases,
            args.phase,
            geom=args.geom,
            full=args.tiles,
            axes_raw=args.axes,
            reps=args.reps,
            iters=args.iters,
            allow_drift=args.allow_drift,
            timeout=args.timeout,
            allow_structural=args.allow_structural,
        )
    if args.cmd == "_one":
        values = [int(v) for v in re.findall(r"-?\d+", args.values)]
        return _one(
            args.cases,
            args.phase,
            geom=args.geom,
            values=values,
            reps=args.reps,
            iters=args.iters,
            src_dir=args.src_dir,
        )
    if args.cmd == "apply":
        values = tuple(int(v) for v in re.findall(r"-?\d+", args.tiles))
        return apply(args.phase, values, geom=args.geom, dry_run=args.dry_run)
    if args.cmd == "prune":
        return _prune()
    raise AssertionError(args.cmd)  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return _dispatch(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
