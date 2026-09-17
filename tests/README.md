# TyBoK Regression Tests (`tests/`)

[English](README.md) | [简体中文](README.zh-CN.md)

Two levels, one entry point:

```bash
cd TyBoK

# checkpoint path: copy tests/checkpoints.env.example to tests/checkpoints.env and fill it in
# (or export TYBOK_CHECKPOINT_<MODEL>=...)
python tests/run_tests.py                       # all models, quick (run it after each change)
python tests/run_tests.py --models fastwam      # only one model
python tests/run_tests.py --level full          # full regression before commit / in CI
python tests/run_tests.py --dry-run             # only print the commands to be run (orchestration visible without a GPU)
```

An individual check can be run directly, with isomorphic arguments (`--level` / `--device` / `--checkpoint` / `--list-jobs`):

```bash
python tests/fastwam/graph_replay.py --level full --replays 200
python tests/smolvla/graph_flags.py --list-jobs
```

**`tests/` depends only on the `tybok` package + torch/numpy**, nothing else.

## Directory

```
tests/
├── run_tests.py              sole entry point: --models / --level / --device / --checkpoint / --dry-run
├── README.md                 this file (English)
├── README.zh-CN.md           Chinese version
├── checkpoints.env.example   local checkpoint path template (copy to checkpoints.env, not tracked)
├── _common/                  shared layer (shared between checks, contains no model knowledge)
│   ├── level.py              Level (quick / full) + filtering jobs by level
│   ├── rows.py               child process → driver result protocol (one ROW JSON line)
│   ├── process.py            one process per job, streaming echo, result collection
│   ├── cli.py                shared CLI arguments, ModelSpec, checkpoint env var/env file, missing-environment decision
│   ├── report.py             declarative table, final RESULT line, exit code
│   ├── overlap_matrix.py     4-mode matrix driver shared by pi05 / smolvla
│   ├── graph_cameras.py      camera-count shape bucket driver shared by the three backends
│   └── compile_modes.py      --compile combination driver shared by the three backends
├── cli/
│   ├── expected.json         the CLI surface golden (regenerate: surface.py --update-expected)
│   └── surface.py            options / defaults / dispatch + parent -> child flag wiring + smoke
├── fastwam/
│   ├── spec.py               checkpoint defaults + kernel tiers
│   ├── graph_replay.py       idempotence + bit-exactness of the sequential kernel / overlap kernel
│   ├── overlap_matrix.py     kernel tiers × graph modes (6 rows)
│   ├── graph_cameras.py      camera-count shape buckets (fastwam: no-op)
│   ├── compile_modes.py      --compile three combinations + whether the compiled regions are real
│   └── sweep_geom.py         pure-logic self-check of kernels/sweep.py (CPU)
├── pi05/
│   ├── spec.py
│   ├── overlap_matrix.py     4 modes
│   ├── graph_cameras.py      camera-count shape buckets (2 graphs)
│   └── compile_modes.py      --compile three combinations + whether the compiled regions are real
└── smolvla/
    ├── spec.py
    ├── overlap_matrix.py     4 modes
    ├── graph_flags.py        per-flag × graph bit-exactness
    ├── graph_cameras.py      camera-count shape buckets (2 and 3)
    └── compile_modes.py      --compile three combinations + whether the compiled regions are real
```

## Two levels: `quick` / `full`

Both levels assert the **same set of invariants** (bit-exactness, overlap actually taking effect, idempotence, compiled regions actually taking effect); the difference is only in how many configurations and how many replays are run, so `quick` passing only means "the configurations it ran pass". The level policy has two granularities, both written in code, not in documentation:

* **Which checks run at `quick`** — the `CHECKS` table in `run_tests.py` (`Check.levels`); checks that do not take part are explicitly listed below the summary table as "not run at quick level" during a quick run, never silently hidden;
* **Which jobs each check runs** — the `quick` markers on that check's own job list, queryable at any time with `--list-jobs`.

| Check | `quick` (daily) | `full` (regression) |
|---|---|---|
| `cli/surface` | 5 guards: options/defaults/dispatch, invariants, parent→child wiring, smoke, self-test | same |
| `fastwam/graph_replay` | sequential kernel + overlap kernel, `--replays 5` | both kernels, `--replays 20` (production value) |
| `fastwam/overlap_matrix` | 1 row: production configuration `graph \| fused coop` | 6 rows: torch/fused-split/fused-coop × eager/graph |
| `fastwam/graph_cameras` | — (full only) | camera-count shape buckets: eager reference + `--graph-cameras 2,3` |
| `fastwam/compile_modes` | — (full only) | 3 `--compile` combinations + whether the compiled regions are real |
| `fastwam/sweep_geom` | 4 pure-logic groups (CPU, a few seconds) | same |
| `pi05/graph_cameras` | eager reference + `--graph-cameras 2,3` | same |
| `pi05/overlap_matrix` | 1 row: `fused+graph` | 4 rows: eager/eager+graph/fused/fused+graph |
| `pi05/compile_modes` | — (full only) | 3 `--compile` combinations |
| `smolvla/graph_cameras` | eager reference + `--graph-cameras 2,3` | same |
| `smolvla/overlap_matrix` | 1 row: `fused+graph` | 4 rows |
| `smolvla/graph_flags` | 2 rows: `baseline` / `no-kv-cache` | 9 rows + `select_action` |
| `smolvla/compile_modes` | — (full only) | 3 `--compile` combinations |

The `quick` trade-off is "each backend runs only the **production configuration** path + its reference": `quick` only answers "did the change I just made break the production configuration"; non-production tiers (torch tier, split form), non-default flags, camera-count shape buckets, `--compile` combinations, and higher replay counts are all left to `full`. The whole `--compile` family is `full`-only: compilation eats the inductor cache anyway (fastwam cold start 137 s), which is a "before commit / CI"-magnitude cost. Measured timing (measured on a single machine, ±20% is normal; the compile rows are numbers with a **warm inductor cache**; the **first fastwam check** of each round also pays for one fused-kernel CUDA extension build, and that row is noticeably higher when the extension cache is cold):

| Check | `quick` | `full` |
|---|---|---|
| `cli/surface` | 19 s | 19 s |
| `fastwam/graph_replay` | 109 s¹ | 98 s |
| `fastwam/overlap_matrix` | 44 s | 261 s |
| `fastwam/graph_cameras` | — | 233 s¹ |
| `fastwam/compile_modes` | — | 258 s |
| `fastwam/sweep_geom` | 6 s | 7 s |
| `pi05/graph_cameras` | 52 s | 38 s |
| `pi05/overlap_matrix` | 38 s | 143 s |
| `pi05/compile_modes` | — | 97 s |
| `smolvla/graph_cameras` | 12 s | 11 s |
| `smolvla/overlap_matrix` | 11 s | 39 s |
| `smolvla/graph_flags` | 19 s | 96 s |
| `smolvla/compile_modes` | — | 38 s |
| **Total (9 of 13 checks)** | **5 m 10 s** | **22 m 17 s** |

¹ The first fastwam check of that round: includes one fused-kernel CUDA extension build (about 88 s / 134 s with a warm extension cache).

`--list-jobs` prints the jobs that will actually run at the current level (the index is the internal `--job-index`), so "what quick actually ran" is queryable at any time, without relying on documentation.

## CI

`.github/workflows/ci.yml` runs three jobs on every push (and on demand through `workflow_dispatch`):

| Job | What it runs | Dependencies |
|---|---|---|
| `lint` | `ruff check .` + `ruff format --check .` (rules, line width and the formatter exemptions come from `[tool.ruff]` / `[tool.ruff.format]` in `pyproject.toml`; the ruff version is pinned in the workflow) | ruff |
| `cli` | `python -m tybok --help` / `models`, the pure-Python import guard on `tybok.policies.fastwam.kernels`, then `python tests/cli/surface.py` | none |
| `regression` | `python tests/run_tests.py --level quick` | torch (CPU wheel) + `[gateway]` |

The `cli` job deliberately installs nothing: the CLI and the backend registration are lazy, so the surface guard runs without numpy / torch / aiohttp, and a subcommand that starts importing the inference stack fails that job. It imports fastwam's pure-Python `kernels/` tooling for the same reason -- that is the code the CPU-only `regression` job drives through `tests/fastwam/sweep_geom.py`, so a backend package that starts eagerly importing its engine fails here first. `regression` reports `SKIP` for every check whose checkpoint or GPU is missing and still exits 0, so a bare CPU runner is a meaningful (if partial) gate -- add `--require-gpu` only on a runner that must really run a GPU.

The same two ruff commands are available locally through `.pre-commit-config.yaml` (which also *fixes* in place instead of failing): `pip install pre-commit && pre-commit install`.

## Conventions

- **One process per job**: a large workspace per kernel tier is resident and a captured graph carries its own private memory pool, so repeatedly rebuilding the engine in the same process OOMs (row 4 of the fastwam matrix is the example). The driver script therefore spawns itself, and the child process selects its job with `--job-index N`.
- **Result protocol**: the child process prints one `ROW {json}` line per job, with the fixed fields `key / status / detail / job / metrics / values`: `job` is the identity of "what ran" (the driver pairs a row with its reference process using it), `metrics` is "what was measured" (the driver builds the table from it), and `values` is the optional raw output vector (used only for rows that need two processes compared to be judged).
  If the child process prints no row (crash, OOM, killed) → the driver synthesizes a `FAIL` row and includes the child process's last output line; it **can never be taken as a pass**.
- **Judgement and exit code**: any row `FAIL` → `RESULT: FAIL`, exit 1; all `SKIP` (no GPU / no checkpoint) → `RESULT: SKIP`, exit 0; otherwise `RESULT: PASS`, exit 0. Usage errors exit 2. A CPU-only CI can therefore run the same command, and "nothing ran" is an explicit `SKIP` in the table. `--require-gpu` turns "missing environment" from `SKIP` into `FAIL` (for machines that must really run on a GPU).
- **The shared layer holds only shared things**: model knowledge (checkpoints, kernel tiers, flag combinations) stays under `<model>/`; `_common/` knows no model.
- **Model-independent checks**: a check that belongs to no backend lives under `tests/<group>/` (today `tests/cli/`, the guard on the command line of `python -m tybok` itself) and is registered with `uses_engine=False`. `--models <group>` selects it like a backend, but `--device` / `--checkpoint` / `--require-gpu` are not passed and it never `SKIP`s for a missing checkpoint -- the regression checks never parse a command line, so this is the only guard on that surface.
- **Single entry point**: `run_tests.py` only selects models/levels, runs serially, and summarizes; each check knows for itself what to run.

## Why only smolvla has `graph_flags`

This check was not "added for symmetry while we were at it"; it guards a risk unique to smolvla:

1. **The switch lives in the captured kernel and does not enter the graph key.** smolvla captures the whole kernel `VLAFlowMatching.sample_actions`, and the graph key is only `(n_cams,)`; yet `sampler == "heun"`, the loop length of `num_steps`, and `if self.cache_expert_prefix_kv:` are all inside the graph body. pi05 / fastwam support only euler (the engine raises directly for non-euler), and fastwam even puts `steps` into the graph key.
2. **Mutable runtime state entered the captured kernel.** `cache_expert_prefix_kv` is per-request mutable state in smolvla, and `--compile` temporarily turns it off for capture and then restores it — "the rule at capture time ≠ the rule at run time", exactly the breeding ground for this class of bug.
3. **History**: on 2026-09-14 there was first smolvla's "step-0 expert cross-attn silently falls back to the eager kernel in the overlap pipeline", then the same class of problem with pi05 `--pad-free`, so an audit of "does smolvla have the same gap" was done (conclusion: no gap, the gate stayed as a regression lock).
4. **The symmetric surface already exists**: the overlap matrices of the three backends already cross `fused tier × graph`; what was not covered is exactly those smolvla flags, so the gate belongs only here, rather than adding an idle version to pi05/fastwam.

(Two known inconsistencies are recorded along the way, neither affecting this set of gates: `cache_expert_prefix_kv` in `pi05/engine.py` is currently a dead parameter, and on pi05 `--no-expert-prefix-kv-cache` is a silent no-op; fastwam raises explicitly for the same parameter.)

## Adding a new model / new check

1. `tests/<model>/spec.py`: a `ModelSpec` (`key` / `checkpoint` / `fused_flags`).
2. Reuse a shared driver (`_common/overlap_matrix.py`, `_common/graph_cameras.py`, `_common/compile_modes.py`) or write it following `fastwam/graph_replay.py`: define a job list (with `quick` markers) + a `_run_job` (child-process side) + a `judge` (driver side) → `print_table` + `finish`.
3. Register it in `CHECKS` in `tests/run_tests.py` (declaring along the way which levels it takes part in), and write down its quick/full coverage in the tables in this file. A check that belongs to no model goes under `tests/<group>/` (see `tests/cli/`) and is registered with `uses_engine=False`.

## Known boundaries

- **No timeout**: a stuck job waits forever (consistent with the old scripts); if you really want to guard against hangs, leave it to the outer CI timeout or `timeout(1)`.
- **Checkpoints do not enter the repository**: by default `$TYBOK_CHECKPOINT_<MODEL>` is used (you may copy `tests/checkpoints.env.example` to `tests/checkpoints.env`, loaded automatically in-process); override with `--checkpoint MODEL=PATH` for the entry point and `--checkpoint PATH` for an individual check.
- **`quick` is not full evidence**: the configurations and checks it runs are in the table above (you can also use `--dry-run` to see the actual commands); the remaining configurations are covered only by `--level full`.
