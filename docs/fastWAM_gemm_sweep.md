# fastWAM GEMM parameter sweep (geometry sweep)

[English](fastWAM_gemm_sweep.md) | [简体中文](fastWAM_gemm_sweep.zh-CN.md)

`tybok/policies/fastwam/kernels/sweep.py` is the only sweep entry point for the fused kernel **geometry** (GEMM tile
parameters): it chains "edit the macro → compile a variant → measure with real tensors → cross-check the bit pattern →
write the macro back" into one pipeline, and it never **touches the production source**.

Prerequisite: the geometry is **compile-time** (the shapes are `constexpr`, and the GEMM's 29 tile parameters are
template parameters), so "changing the geometry" = editing the `FWAM_<KERNEL>_<PHASE>_TILES` macro + recompiling, not
a runtime switch. The sweep always targets the parameter set that `PhaseSpec.tiles` points at, never a rewritten
kernel.

Tool file map:

| File | Responsibility |
| --- | --- |
| `kernels/sweep.py` | Sweep entry point (the main subject of this document): `list` / `capture` / `run` / `apply` / `prune` |
| `kernels/phases.py` | Phase registry: 41 phases (`PHASES`), the per-kernel IO contract and `SMEM` (`KERNEL_IO`), `dispatch` |
| `kernels/phase_check.py` | Record/replay/per-buffer bit-pattern comparison (`Recorder` / `diff_buffer`) |
| `kernels/geometry.py` | **Declares** one geometry per architecture (`(major, minor)`) + cross-checks it against the binary (`check` / `--bootstrap`) |
| `kernels/vdit_gemm_core.cu` / `tmt5_gemm_core.cu` | Shared GEMM body — the 29 axes are its template parameters |

## 1. Coverage

8 kernels with 41 phases in total, of which **10 phases** carry sweepable geometry (i.e. the ones that go through the shared GEMM body):

| Phase | Geometry label (= geometry table key = `fwam_tiles` self-reported name) | Macro | Source file |
| --- | --- | --- | --- |
| `vdit.attn_self/P2`, `P6` | `S_PHASE_2`, `S_PHASE_6` | `FWAM_VDIT_SELF_S_PHASE_{2,6}_TILES` | `vdit_attn_self.cu` |
| `vdit.attn_cross/P2`, `P6` | `C_PHASE_2_2G`, `C_PHASE_6` | `FWAM_VDIT_CROSS_C_PHASE_{2_2G,6}_TILES` | `vdit_attn_cross.cu` |
| `vdit.ffn/P2`, `P4` | `F_PHASE_2`, `F_PHASE_4` | `FWAM_VDIT_FFN_F_PHASE_{2,4}_TILES` | `vdit_ffn.cu` |
| `tmt5.attn/P2`, `P5` | `S_PHASE_2`, `S_PHASE_5` | `FWAM_TMT5_ATTN_S_PHASE_{2,5}_TILES` | `tmt5_attn.cu` |
| `tmt5.ffn/P2`, `P4` | `F_PHASE_2`, `F_PHASE_4` | `FWAM_TMT5_FFN_F_PHASE_{2,4}_TILES` | `tmt5_ffn.cu` |

- Every phase currently declares only one geometry label, so `--geom` is generally unnecessary; `--geom` is for
  choosing one when "a phase declares several geometries" (it is required when `PhaseSpec.tiles` has more than one
  entry).
- **`adit.*` (the action's 16 phases) is not part of this scheme**: their geometry is a hand-written `constexpr` and
  does not go through the shared GEMM body, so changing it means changing the source.
- The `SMEM` budget comes from the phase table (`KERNEL_IO[kernel].geometry["SMEM"]`): `vdit.*` = 49152 B (48 KB),
  `tmt5.*` = 98304 B (96 KB). The sweep uses it for an early warning about an estimated over-budget case.

## 2. The 29 axes

`--tiles` is the **positional argument order** of 29 integers, identical to the template parameter order of
`fwam_fp8_gemm_body` (the axis names are parsed by the script from the source, not transcribed by hand).

| Axis | Semantics | Category |
| --- | --- | --- |
| `BM`, `BN`, `BK` | Tile extent in the M / N / K direction | blocking |
| `SA`, `SB` | Stage depth of the A ring / B(=W) ring (the two are independent; `SB>=SA`) | pipeline |
| `WM`, `WN` | 2D warp partitioning (`NWARPS = WM*WN`; with 8 warps, 4×2 has the least B redundancy) | warp |
| `G2` | Split each stage into two cp.async commits, A/B (triton cadence) | scheduling |
| `LIN` | cp.async stores in dst-linear order (eliminates store bank conflicts) | layout |
| `ORD` | 0 = issue the next stage after mma (measured better), 1 = issue immediately after the barrier | scheduling |
| `BORD` | 0 = A issued first, 1 = B(W) issued first (W is on the DRAM critical path) | scheduling |
| `XS` | triton-style XOR-swizzle smem layout (4 KB window); `XS=2` is dense+pad | layout |
| `LDB` | Double-buffer the ldmatrix fragments (overlaps the LSU with the tensor pipeline) | pipeline |
| `ACG`, `WCG` | Use `.cg` (bypassing L1) instead of `.ca` for the A ring / W ring cp.async | cache |
| `AHINT`, `WHINT` | Whether the A ring / W ring carries `L2::cache_hint` (off = the eviction policy stops working) | cache |
| `EV` | W ring fetch/eviction variant: 0 = `.ca` (measured best on this machine), 1 = `.cg` + W `evict_first` (triton style), 3 = pure `.cg` (control) | cache |
| `WPF` | L2 prefetch granularity of the W ring cp.async (0 / 128 / 256) | cache |
| `PH` | partial / `Cout` stores carry `L2::evict_first` (stored data does not steal W/A's L2 ways) | cache |
| `SPLIT` | split-K: when >1, write fp32 partials + a per-tile atomic counter, and the last block reduces (no second launch) | split-K |
| `PD` | In split mode, write the partial directly (skip smem staging and one barrier) | split-K |
| `F16P` | Use fp16 for the partial (halves the traffic; requires the partial sums to fall within fp16 range) | split-K |
| `EPI` | Fused epilogue: `u=bf16RN(acc*sa*sw+bias)` → `g=bf16RN(gelu(u))` → write gbuf + per-row `|g|` atomicMax (only with `SPLIT=1`) | epilogue |
| `RESID` | Append the FFN residual at the end of the epilogue (requires `ffn_ex`) | epilogue |
| `FA` | In-stage fp16 accumulation (on sm_89 fp8+f16acc = 2× tensor throughput), promoted to fp32 at the end of the stage; requires each stage's partial sums to fall within fp16 range | numeric tier |
| `FAP` | The promotion period of `FA` (only meaningful when `FA` is on) | numeric tier |
| `APATH` | A does not use cp.async: LDG→register (1 stage ahead)→STS; B still uses cp.async | fetch path |
| `PERSIST` | Change the grid to 1D and have each CTA grid-stride serially over several (m,n) tiles (tile linearization `t = m + n*MT`, so that two m-tiles of the same `n` land on adjacent CTAs → the W tile is shared in L2) | grid |

### Compile-time constraints (why some candidates "failed to compile")

The `static_assert`s in the source reject illegal combinations outright; the sweep records such candidates as one
`invalid` row:

- `SA >= 2 && SB >= 2 && BK % 32 == 0 && BM % (16*WM) == 0 && BN % (8*WN) == 0`
- `APATH` is only implemented under `LIN && !G2`; `EPI` requires `SPLIT == 1`; `PD` requires `SPLIT > 1`
- `XS` requires `BK % 64 == 0 && BM % 64 == 0`

### Axes that cannot be swept by "editing the macro only" (`_STRUCTURAL`)

These 9 axes change the **contract between the kernel and the host**, not just the blocking. The script rejects them
by default (`--allow-structural` forces a compile, but editing the macro alone does not amount to a correct
implementation — the source must be changed as well):

| Axis | Why editing the macro alone is not enough |
| --- | --- |
| `PERSIST` | Changes the grid shape (`T0_ = PERSIST ? blockIdx.x : blockIdx.x + blockIdx.y*MT_`): the persistent form relies on a 1D grid-stride, while the non-persistent form needs a 2D tile grid, and the host-side `*_split_grid()` is computed for the persistent form → editing the macro alone = using a 1D grid as if it were 2D, with out-of-bounds writes (measured: the whole output is NaN) |
| `SPLIT` | Changes the partial output path (whether a partial buffer is needed, and who reduces it) |
| `PD` | In split mode, write the partial directly (one fewer smem staging and one fewer barrier) |
| `EPI` | The fused epilogue writes gbuf/raw1 (only implemented with `SPLIT=1`) |
| `RESID` | The epilogue appends a residual term (changes the output sink) |
| `APATH` | A goes LDG→reg→STS (a different issue path and a different register staging) |
| `FA` | In-stage fp16 accumulation — a **numeric tier change**; production-scale activations overflow to NaN |
| `F16P` | fp16 partials (same as above, overflow risk) |
| `FAP` | The promotion period of `FA` (only meaningful when `FA` is on) |

### Example: reading a phase's current values

```bash
python -m tybok.policies.fastwam.kernels.sweep list --phase vdit.ffn/P4
# vdit.ffn/P4  geometry F_PHASE_4  macro FWAM_VDIT_FFN_F_PHASE_4_TILES (vdit_ffn.cu)
#     BM      = 64
#     BN      = 48
#     BK      = 128
#     SA      = 3
#     SB      = 3
#     WM      = 4
#     WN      = 2
#     SPLIT   = 1
#     ...(29 rows in total, including XS=2, PERSIST=1, RESID=1)
```

## 3. Subcommands

| Subcommand | What it does | Key arguments (defaults) |
| --- | --- | --- |
| `list` | List every sweepable geometry; with `--phase`, print that phase's axis names and current values (positional order = `--tiles` order) | `--phase` |
| `capture` | Run one **real inference** (split-phase form) and use `phase_check.Recorder` to record each phase's entry state as a case | `--model` (required), `--out` (required), `--kernels`, `--per-phase` (1), `--cameras`, `--steps`, `--sampler` (euler), `--seed` (0), `--text-encoder-device` (**cuda**), `--coop` |
| `run` | Sweep a candidate set: bit-pattern cross-check + interleaved A/B | `<cases>`, `--phase` (required), `--axes "BN=32,48,64 BK=128"`, `--tiles` (a full 29-tuple, repeatable), `--reps` (5), `--iters` (100), `--allow-drift`, `--allow-structural`, `--timeout` (600 s/candidate) |
| `_one` | Internal: measure **one** candidate in a subprocess and print one line of JSON (normally not typed by hand) | `<cases>`, `--phase`, `--values`, `--src-dir` (omitted = measure the production extension itself = the baseline row) |
| `apply` | Write the chosen geometry back into the production source macro (only the numbers change, in place) | `--phase`, `--tiles`, `--geom`, `--dry-run` |
| `prune` | Delete the sweep scratch and every variant extension build | — |

## 4. Acceptance criteria (both must hold for a candidate to be usable)

**① Bit pattern** — compare the exit state at recording time buffer by buffer with `torch.equal`. When blocking
parameters change (`BN`/`BK`/stage count/warp partitioning) the accumulation order over k does not change, so the
result **should stay bit-exact**; a reported difference can only mean one of two things: that axis inherently changes
the numerics (`SPLIT`/`FA`/`F16P`/`FAP`/`PERSIST`/`PD`/`EPI`/`RESID`), or the computation is wrong. Neither of the two should be
adopted "silently", so by default only bit-exact candidates are recommended; changing the numerics requires an explicit `--allow-drift`.

**② Finiteness** — any inf/nan in the output is disqualified on the spot: `FA=1` (in-stage fp16 accumulation) runs
perfectly fine on small activations, but at production-scale activations it turns into NaN across the board.

**Timing conventions** (three columns in the report):

- the `candidate` column lists only the **axes that differ from the current values** (`= current geometry` means it is the current geometry, i.e. the baseline);
- the `bit pattern` column is the verdict; `min us` is that implementation's own absolute time cost; `rel` is the ratio against the **baseline re-measured in the same process**
  — absolute values drift by a few percent across processes, so **only `rel` is comparable**;
- if `<- not adopted` appears at the end of a row, that candidate is unusable (failed to compile / crashed / not finite / changes the numerics without
  `--allow-drift`).

Two self-checks: **the baseline is always the first row** (current geometry vs current geometry, `rel` should be ≈ `1.000` — this is the timing method's own self-check);
if in some subprocess **the baseline itself** fails to match the recorded exit (`prod_ok` is false), the script warns that "this round's numbers are untrustworthy" — first check whether
this phase is deterministic (in the registry `reduction` should be fixed).

## 5. A complete sweep

```bash
cd TyBoK

# 0) Record the phase entries of one real inference (split-phase form; one recording covers all 41 phases)
python -m tybok.policies.fastwam.kernels.sweep capture \
    --model /path/to/fastwam_checkpoint --out /tmp/cases.pt

# 1) See which axes this phase has and what their current values are
python -m tybok.policies.fastwam.kernels.sweep list --phase vdit.ffn/P4

# 2) Sweep: give only the axes to change (the rest keep the current macro values); candidates = current geometry + the Cartesian product of those axes
python -m tybok.policies.fastwam.kernels.sweep run /tmp/cases.pt \
    --phase vdit.ffn/P4 --axes "BN=32,48,64 BK=64,128"
#    The full 29-tuple can also be given directly (repeatable): --tiles "64 48 128 3 3 4 2 1 ..."

# 3) Adopt the apply command printed at the end of run (look at the diff with --dry-run first)
python -m tybok.policies.fastwam.kernels.sweep apply --phase vdit.ffn/P4 \
    --tiles "…29 values…" --dry-run

# 4) After writing back (do not reorder): recompile → reconcile → regression
python -m tybok.policies.fastwam.kernels.geometry          # cross-check: reports which entry disagrees with the binary
python -m tybok.policies.fastwam.kernels.geometry --bootstrap   # only when the table needs to be rebuilt
python tests/run_tests.py --models fastwam                  # three-family bit-exactness gate (see tests/README.md)

# 5) Reclaim the sweep scratch and the variant extensions
python -m tybok.policies.fastwam.kernels.sweep prune
```

`run` only prints "the fastest usable candidate" at the end; if that candidate is the current geometry, it says
straight out "nothing better, no need to touch the macro".

## 6. Three hard design rules

1. **One candidate, one subprocess**. A bad geometry does not fail gently: after exceeding the tier smem budget the kernel keeps running and writes out of bounds
   (reporting illegal memory access), and at that point **the entire CUDA context is already dead**, so every remaining candidate in the same process is invalidated.
   A crash/timeout only records one row (`invalid: ...`) and then moves on to the next one.
2. **Take the min, not the median**; **do not record graphs inside the measurement loop**. Replaying one at a time jitters more than the difference you are
   looking for, and `min` is the robust estimate for a microbenchmark (it corresponds to "the round that was not disturbed"); capture takes tens of milliseconds and perturbs
   state, so each implementation records one CUDA graph first and afterwards only replays are alternated, swapping the order on odd/even rounds.
3. **Do not touch the production source + name by content hash**. The whole `kernels/` tree is copied into scratch (`$FASTWAM_SWEEP_DIR`, default
   `~/.cache/fastwam_geom_sweep/<content hash>/`), and only the numbers of the target macro are swapped in place (commas, spaces and line-continuation positions are all left alone);
   if the macro was not actually changed (content hash identical to production) it errors out on the spot — otherwise every number the sweep produces would be fake; the same geometry is compiled only once.
   The `smem` estimate is just an early warning (`row = BK + (16 if XS==2 else 0)`, `est = BM*row*SA + BN*row*SB`);
   the kernel itself is authoritative (overestimate = out-of-bounds crash, underestimate = the subprocess catches it too).

## 7. Things to watch out for

- **Sweepable only in the split-phase form**: the cooperative form is a single launch and has no phases (`--coop` only affects the recording content, it cannot be swept).
- **`capture` defaults to `--text-encoder-device cuda`** (unlike the worker default of `cpu`): when UMT5 stays on the CPU,
  text fusion is rejected by the engine and not a single one of the 9 `tmt5` phases can be recorded; one recording has to cover all 41 phases, so the default here puts it on the GPU
  (the embedding table still stays on the CPU).
- **Switching GPUs/architectures means re-running and re-reconciling**: the geometry table is declared per `(major, minor)` (e.g. `(8, 9)` = sm_89); an architecture that is not in
  the table is reported truthfully as "no such architecture in the table", rather than having a value made up for it.
- **Cold compile cost**: every new geometry requires compiling a variant CUDA extension, which is noticeably expensive when sweeping a large grid; after `prune` you can start over from scratch.
- Candidates from `--allow-drift` may be on a different numeric tier than production (the drift tier), so always run the regression suite before adopting one; `--allow-structural`
  only "forces a compile", it does not mean the result is correct.
