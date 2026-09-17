# TyBoK 回归测试（`tests/`）

[English](README.md) | [简体中文](README.zh-CN.md)

两级、一个入口：

```bash
cd TyBoK

# checkpoint 路径：复制 tests/checkpoints.env.example 为 tests/checkpoints.env 并填好
#（或 export TYBOK_CHECKPOINT_<MODEL>=...）
python tests/run_tests.py                       # 全部模型，quick（改完随手跑）
python tests/run_tests.py --models fastwam      # 只看一个模型
python tests/run_tests.py --level full          # 提交前/CI 的完整回归
python tests/run_tests.py --dry-run             # 只打印将要执行的命令（无 GPU 也能看编排）
```

单项检查可以直接跑，参数同构（`--level` / `--device` / `--checkpoint` / `--list-jobs`）：

```bash
python tests/fastwam/graph_replay.py --level full --replays 200
python tests/smolvla/graph_flags.py --list-jobs
```

**`tests/` 只依赖 `tybok` 包 + torch/numpy**，没有其他依赖。

## 目录

```
tests/
├── run_tests.py              唯一入口：--models / --level / --device / --checkpoint / --dry-run
├── README.md                 英文版（English）
├── README.zh-CN.md           本文件（中文）
├── checkpoints.env.example   本机 checkpoint 路径模板（复制为 checkpoints.env，不入库）
├── _common/                  公共层（检查之间共享，不含模型知识）
│   ├── level.py              Level（quick / full）+ 按级筛 job
│   ├── rows.py               子进程 → 驱动的结果协议（一行 ROW JSON）
│   ├── process.py            一个 job 一个进程、流式回显、收集结果
│   ├── cli.py                公共命令行参数、ModelSpec、checkpoint 环境变量/环境文件、环境缺失判定
│   ├── report.py             声明式表格、末行 RESULT、退出码
│   ├── overlap_matrix.py     pi05 / smolvla 共用的 4 模式矩阵驱动
│   ├── graph_cameras.py      三个后端共用的相机数形状桶驱动
│   └── compile_modes.py      三个后端共用的 --compile 组合驱动
├── cli/
│   ├── expected.json         CLI 表层的金标（重生成：surface.py --update-expected）
│   └── surface.py            选项/默认值/派发 + 父子进程 flag 接线 + 烟测
├── fastwam/
│   ├── spec.py               checkpoint 默认值 + kernel 档位
│   ├── graph_replay.py       顺序核 / overlap 核的幂等 + 逐位一致
│   ├── overlap_matrix.py     kernel 档 × graph 模式（6 行）
│   ├── graph_cameras.py      相机数形状桶（fastwam：no-op）
│   ├── compile_modes.py      --compile 三组合 + 编译区域真伪
│   └── sweep_geom.py         kernels/sweep.py 的纯逻辑自检（CPU）
├── pi05/
│   ├── spec.py
│   ├── overlap_matrix.py     4 模式
│   ├── graph_cameras.py      相机数形状桶（2 个图）
│   └── compile_modes.py      --compile 三组合 + 编译区域真伪
└── smolvla/
    ├── spec.py
    ├── overlap_matrix.py     4 模式
    ├── graph_flags.py        逐 flag × graph 逐位一致
    ├── graph_cameras.py      相机数形状桶（2 和 3）
    └── compile_modes.py      --compile 三组合 + 编译区域真伪
```

## 两级：`quick` / `full`

两级断言的是**同一批不变量**（逐位一致、overlap 真的生效、幂等、编译区域真的生效），差别只在跑多少配置、多少次 replay，
所以 `quick` 通过只说明「它跑过的那些配置通」。级别政策有两个粒度，都写在代码里、不靠文档：

* **哪些检查进 `quick`** —— `run_tests.py` 的 `CHECKS` 表（`Check.levels`）；不进的检查在 quick 运行时会在汇总表下方
  显式列出「not run at quick level」，不会静默隐藏；
* **每个检查跑哪些 job** —— 该检查自己的 job 列表上的 `quick` 标记，`--list-jobs` 随时可查。

| 检查 | `quick`（日常） | `full`（回归） |
|---|---|---|
| `cli/surface` | 5 道守卫：选项/默认值/派发、不变量、父子接线、烟测、自检 | 同 |
| `fastwam/graph_replay` | 顺序核 + overlap 核，`--replays 5` | 双核，`--replays 20`（生产值） |
| `fastwam/overlap_matrix` | 1 行：生产档 `graph \| fused coop` | 6 行：torch/fused-split/fused-coop × eager/graph |
| `fastwam/graph_cameras` | —（full only） | 相机数形状桶：eager 参照 + `--graph-cameras 2,3` |
| `fastwam/compile_modes` | —（full only） | 3 个 `--compile` 组合 + 编译区域真伪 |
| `fastwam/sweep_geom` | 4 组纯逻辑（CPU，几秒） | 同 |
| `pi05/graph_cameras` | eager 参照 + `--graph-cameras 2,3` | 同 |
| `pi05/overlap_matrix` | 1 行：`fused+graph` | 4 行：eager/eager+graph/fused/fused+graph |
| `pi05/compile_modes` | —（full only） | 3 个 `--compile` 组合 |
| `smolvla/graph_cameras` | eager 参照 + `--graph-cameras 2,3` | 同 |
| `smolvla/overlap_matrix` | 1 行：`fused+graph` | 4 行 |
| `smolvla/graph_flags` | 2 行：`baseline` / `no-kv-cache` | 9 行 + `select_action` |
| `smolvla/compile_modes` | —（full only） | 3 个 `--compile` 组合 |

`quick` 的取舍是「每个后端只跑**生产档**那条路径 + 它的参照」：`quick` 只回答「刚才这一改有没有把生产配置弄坏」；
非生产档（torch 档、split 形式）、非默认 flag、相机数形状桶、`--compile` 组合、以及更高的 replay 次数都交给 `full`。
`--compile` 整族只在 `full`：编译本来就要吃 inductor 缓存（fastwam 冷启动 137 s），是「提交前/CI」量级的成本。
实测耗时（单机实测，±20% 属正常；compile 行是 **inductor 缓存已热**的数字；每轮**第一个 fastwam
检查**还要付一次融合核 CUDA 扩展的构建，扩展缓存冷时该行明显偏高）：

| 检查 | `quick` | `full` |
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
| **合计（9 项 / 13 项检查）** | **5 m 10 s** | **22 m 17 s** |

¹ 该轮的第一个 fastwam 检查：含一次融合核 CUDA 扩展构建（扩展缓存已热时约 88 s / 134 s）。

`--list-jobs` 打出当前级别实际会跑的 job（下标即内部 `--job-index`），所以「quick 到底跑了什么」是随时可查的，
不靠文档。

## CI

`.github/workflows/ci.yml` 在每次 push（以及手动 `workflow_dispatch`）时跑三个 job：

| Job | 跑什么 | 依赖 |
|---|---|---|
| `lint` | `ruff check .` + `ruff format --check .`（规则、行宽与 formatter 豁免都取自 `pyproject.toml` 的 `[tool.ruff]` / `[tool.ruff.format]`；ruff 版本在 workflow 里钉死） | ruff |
| `cli` | `python -m tybok --help` / `models`，再跑 `python tests/cli/surface.py` | 无 |
| `regression` | `python tests/run_tests.py --level quick` | torch（CPU 轮子）+ `[gateway]` |

`cli` 这个 job 故意什么都不装：CLI 与后端注册都是惰性的，所以 CLI 表面守卫不需要 numpy / torch / aiohttp，
哪个子命令开始导入推理栈就会让这个 job 红。`regression` 对缺 checkpoint / 缺 GPU 的检查报 `SKIP` 且仍退出 0，
所以纯 CPU 机器上的 CI 是有意义（虽然不完整）的门——`--require-gpu` 只加在必须真跑 GPU 的机器上。

同样这两条 ruff 命令在本地可以通过 `.pre-commit-config.yaml` 跑（本地是就地修，而不是报错）：
`pip install pre-commit && pre-commit install`。

## 约定

- **一个 job 一个进程**：分档 kernel 常驻大工作区、被捕获的 graph 自带私有显存池，同进程反复重建引擎会 OOM
  （fastwam 矩阵第 4 行就是例子）。所以驱动脚本自己 spawn 自己，子进程用 `--job-index N` 选 job。
- **结果协议**：子进程每个 job 打一行 `ROW {json}`，字段固定 `key / status / detail / job / metrics / values`：
  `job` 是「跑了什么」的身份（驱动据此把一行和它的参照进程配对），`metrics` 是「量到了什么」（驱动据此出表），
  `values` 是可选的原始输出向量（只有需要两个进程比对才能判定的行才用）。
  子进程没打出行（崩了、OOM、被杀）→ 驱动合成一行 `FAIL` 并带上子进程最后一行输出，**不可能被当成通过**。
- **判定与退出码**：任一行 `FAIL` → `RESULT: FAIL`，退出 1；全部 `SKIP`（没 GPU / 没 checkpoint）→ `RESULT: SKIP`，退出 0；
  否则 `RESULT: PASS`，退出 0。用法错误退出 2。所以 CPU-only 的 CI 可以跑同一条命令，而「什么都没跑」在表上是显式的 `SKIP`。
  `--require-gpu` 把「环境缺失」从 `SKIP` 变成 `FAIL`（给必须真跑 GPU 的机器用）。
- **公共层只放公共的东西**：模型知识（checkpoint、kernel 档、flag 组合）留在 `<model>/` 下；`_common/` 不认识任何模型。
- **不属于任何模型的检查**：放在 `tests/<group>/` 下（目前是 `tests/cli/`，守的就是 `python -m tybok` 自己的命令行），
  登记时带 `uses_engine=False`。`--models <group>` 能像后端一样选中它，但不会传 `--device` / `--checkpoint` / `--require-gpu`，
  也不会因为缺 checkpoint 而 `SKIP`——因为其它回归检查都不解析命令行，这是那一层唯一的守卫。
- **入口唯一**：`run_tests.py` 只负责选模型/级别、串行跑、汇总；每个检查自己知道跑什么。

## 为什么只有 smolvla 有 `graph_flags`

这个检查不是「顺手对称加的」，它守的是 smolvla 独有的风险：

1. **开关落在被捕获的核心里，且不进图键。** smolvla 捕获整核 `VLAFlowMatching.sample_actions`，图键只有
   `(n_cams,)`；而 `sampler == "heun"`、`num_steps` 的循环长度、`if self.cache_expert_prefix_kv:` 都在图体内。
   pi05 / fastwam 只支持 euler（引擎对非 euler 直接抛错），fastwam 还把 `steps` 放进了图键。
2. **运行期可变状态进了被捕获核心。** `cache_expert_prefix_kv` 在 smolvla 是 per-request 可变状态，`--compile`
   会为捕获临时关掉它再恢复——「捕获时的规则 ≠ 运行时的规则」，正是这类 bug 的温床。
3. **历史**：2026-09-14 先有 smolvla 的「step-0 expert cross-attn 在 overlap 流水线里静默回退 eager kernel」，
   再有 pi05 `--pad-free` 的同类问题，于是做了「smolvla 有没有同类缺口」的审计（结论：无缺口，门留下当回归锁）。
4. **对称面已经存在**：三个后端的 overlap 矩阵本身就在做 `fused 档 × graph` 的交叉；没被覆盖的恰好是 smolvla
   那几个 flag，所以只该在这里加门，而不是给 pi05/fastwam 补一个空转的版本。

（顺带记录两个已知不一致，都不影响本套门：`pi05/engine.py` 的 `cache_expert_prefix_kv` 目前是死参数，
pi05 上 `--no-expert-prefix-kv-cache` 是静默 no-op；fastwam 对同一参数是显式报错。）

## 接一个新模型 / 新检查

1. `tests/<model>/spec.py`：一个 `ModelSpec`（`key` / `checkpoint` / `fused_flags`）。
2. 复用共享驱动（`_common/overlap_matrix.py`、`_common/graph_cameras.py`、`_common/compile_modes.py`）或照
   `fastwam/graph_replay.py` 写：定义 job 列表（带 `quick` 标记）+ 一个 `_run_job`（子进程侧）+ 一个 `judge`（驱动侧）
   → `print_table` + `finish`。
3. 在 `tests/run_tests.py` 的 `CHECKS` 里登记（顺便声明它参加哪些级别），并在本文件的表里写下它的 quick/full 覆盖。
   不属于任何模型的检查放在 `tests/<group>/` 下（见 `tests/cli/`），登记时带 `uses_engine=False`。

## 已知边界

- **没有超时**：卡住的 job 会一直等（与旧脚本一致）；真要防挂，交给外层 CI 的超时或 `timeout(1)`。
- **checkpoint 不进仓库**：默认取 `$TYBOK_CHECKPOINT_<MODEL>`（可复制 `tests/checkpoints.env.example` 为 `tests/checkpoints.env`，进程内自动加载）；入口用 `--checkpoint MODEL=PATH`、单项检查用 `--checkpoint PATH` 覆盖。
- **`quick` 不是全量证据**：它跑的配置与检查见上表（也可以用 `--dry-run` 看实际命令）；其余配置只有 `--level full` 才覆盖。
