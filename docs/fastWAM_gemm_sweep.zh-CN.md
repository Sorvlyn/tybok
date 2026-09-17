# fastWAM GEMM 扫参（几何 sweep）

[English](fastWAM_gemm_sweep.md) | [简体中文](fastWAM_gemm_sweep.zh-CN.md)

`tybok/policies/fastwam/kernels/sweep.py` 是融合核**几何**（GEMM tile 参数）的唯一扫参入口：它把
「改宏 → 编译变体 → 用真实张量量测 → 位型对拍 → 写回宏」串成一条链，全程**不动生产源码**。

前提：几何是**编译期**的（形状是 `constexpr`，GEMM 的 29 个 tile 参数是模板参数），所以「改几何」=
改 `FWAM_<KERNEL>_<PHASE>_TILES` 宏 + 重编译，不是运行时开关。扫的永远是 `PhaseSpec.tiles` 指向的
那组参数，不是重写核。

工具文件地图：

| 文件 | 职责 |
| --- | --- |
| `kernels/sweep.py` | 扫参入口（本文档的主角）：`list` / `capture` / `run` / `apply` / `prune` |
| `kernels/phases.py` | 相登记处：41 个相（`PHASES`）、每核的 IO 契约与 `SMEM`（`KERNEL_IO`）、`dispatch` |
| `kernels/phase_check.py` | 录制/回放/逐 buffer 位型比较（`Recorder` / `diff_buffer`） |
| `kernels/geometry.py` | 每架构（`(major, minor)`）**声明**一组几何 + 与二进制对账（`check` / `--bootstrap`） |
| `kernels/vdit_gemm_core.cu` / `tmt5_gemm_core.cu` | 共享 GEMM body —— 29 个轴就是它的模板参数 |

## 1. 覆盖范围

8 个 kernel 共 41 个相，其中 **10 个相**带着可扫几何（即走共享 GEMM body 的那些）：

| 相 | 几何标签（= geometry 表键 = `fwam_tiles` 自报名） | 宏 | 源文件 |
| --- | --- | --- | --- |
| `vdit.attn_self/P2`、`P6` | `S_PHASE_2`、`S_PHASE_6` | `FWAM_VDIT_SELF_S_PHASE_{2,6}_TILES` | `vdit_attn_self.cu` |
| `vdit.attn_cross/P2`、`P6` | `C_PHASE_2_2G`、`C_PHASE_6` | `FWAM_VDIT_CROSS_C_PHASE_{2_2G,6}_TILES` | `vdit_attn_cross.cu` |
| `vdit.ffn/P2`、`P4` | `F_PHASE_2`、`F_PHASE_4` | `FWAM_VDIT_FFN_F_PHASE_{2,4}_TILES` | `vdit_ffn.cu` |
| `tmt5.attn/P2`、`P5` | `S_PHASE_2`、`S_PHASE_5` | `FWAM_TMT5_ATTN_S_PHASE_{2,5}_TILES` | `tmt5_attn.cu` |
| `tmt5.ffn/P2`、`P4` | `F_PHASE_2`、`F_PHASE_4` | `FWAM_TMT5_FFN_F_PHASE_{2,4}_TILES` | `tmt5_ffn.cu` |

- 每个相现在都只挂一个几何标签，所以 `--geom` 一般不用；`--geom` 是给「一个相声明了多个几何」时选
  其一用的（`PhaseSpec.tiles` 有多项时会要求指定）。
- **`adit.*`（action 的 16 个相）不在这套里**：它们的几何直接是手写 `constexpr`，不经共享 GEMM body，
  改它要改源码。
- `SMEM` 预算取自相位表（`KERNEL_IO[kernel].geometry["SMEM"]`）：`vdit.*` = 49152 B（48 KB），
  `tmt5.*` = 98304 B（96 KB）。扫参用它做「估计超预算」的提前告警。

## 2. 29 个轴

`--tiles` 是 29 个整数的**位置参数序**，与 `fwam_fp8_gemm_body` 的模板参数顺序一致（轴名由脚本从
源码里解析，不手抄）。

| 轴 | 语义 | 分类 |
| --- | --- | --- |
| `BM`、`BN`、`BK` | tile 在 M / N / K 方向的尺寸 | 分块 |
| `SA`、`SB` | A 环 / B(=W) 环的 stage 深度（两者独立，`SB>=SA`） | 流水线 |
| `WM`、`WN` | 2D warp 分区（`NWARPS = WM*WN`；8 warp 时 4×2 的 B 冗余最小） | warp |
| `G2` | 每 stage 拆 A/B 两个 cp.async commit（triton 节奏） | 调度 |
| `LIN` | cp.async 写按 dst-线性序（消写 bank 冲突） | 布局 |
| `ORD` | 0 = 下 stage 在 mma 后 issue（实测优），1 = barrier 后立即 issue | 调度 |
| `BORD` | 0 = A 先发，1 = B(W) 先发（W 在 DRAM 关键路径上） | 调度 |
| `XS` | triton 式 XOR-swizzle smem 布局（4 KB 窗）；`XS=2` 为 dense+pad | 布局 |
| `LDB` | ldmatrix 片段双缓冲（LSU 与 tensor 流水重叠） | 流水线 |
| `ACG`、`WCG` | A 环 / W 环的 cp.async 用 `.cg`（绕 L1）而非 `.ca` | 缓存 |
| `AHINT`、`WHINT` | A 环 / W 环是否带 `L2::cache_hint`（关掉 = 逐出策略失效） | 缓存 |
| `EV` | W 环取数/逐出变体：0 = `.ca`（本机实测最优）、1 = `.cg` + W `evict_first`（triton 风格）、3 = `.cg` 纯（对照） | 缓存 |
| `WPF` | W 环 cp.async 的 L2 预取粒度（0 / 128 / 256） | 缓存 |
| `PH` | partial / `Cout` 写带 `L2::evict_first`（写数据不抢 W/A 的 L2 way） | 缓存 |
| `SPLIT` | split-K：>1 时写 fp32 partial + 每 tile atomic 计数器、末 block 归约（无第二个 launch） | split-K |
| `PD` | split 模式直接写 partial（跳过 smem staging 与一次 barrier） | split-K |
| `F16P` | partial 用 fp16（流量减半，要求部分和落在 fp16 范围内） | split-K |
| `EPI` | 融合 epilogue：`u=bf16RN(acc*sa*sw+bias)` → `g=bf16RN(gelu(u))` → 写 gbuf + per-row `|g|` atomicMax（仅 `SPLIT=1`） | epilogue |
| `RESID` | epilogue 末尾追加 FFN 残差（需 `ffn_ex`） | epilogue |
| `FA` | 级内 fp16 累加（sm_89 上 fp8+f16acc = 2× tensor 吞吐），级末促升 fp32；要求每 stage 部分和在 fp16 范围内 | 数值档 |
| `FAP` | `FA` 的促升周期（仅 `FA` 开时有效） | 数值档 |
| `APATH` | A 不走 cp.async：LDG→寄存器（提前 1 stage）→STS；B 仍走 cp.async | 取数路径 |
| `PERSIST` | grid 改成 1D、每个 CTA grid-stride 串行处理多个 (m,n) tile（tile 线性化 `t = m + n*MT`，使同 `n` 的两个 m-tile 落在相邻 CTA → W tile 在 L2 被共享） | grid |

### 编译期约束（为什么有些候选会「failed to compile」）

源码里的 `static_assert` 直接挡住非法组合，扫参把这类候选记成一行 `invalid`：

- `SA >= 2 && SB >= 2 && BK % 32 == 0 && BM % (16*WM) == 0 && BN % (8*WN) == 0`
- `APATH` 只在 `LIN && !G2` 下实现；`EPI` 要求 `SPLIT == 1`；`PD` 要求 `SPLIT > 1`
- `XS` 要求 `BK % 64 == 0 && BM % 64 == 0`

### 不能用「只改宏」扫的轴（`_STRUCTURAL`）

这 9 个轴改的是**核与宿主的契约**，不单是 blocking。脚本默认直接拒绝（`--allow-structural` 可强行
编译，但只改宏并不构成一个正确的实现，需同时改源码）：

| 轴 | 为什么不能只改宏 |
| --- | --- |
| `PERSIST` | 改变 grid 形状（`T0_ = PERSIST ? blockIdx.x : blockIdx.x + blockIdx.y*MT_`）：持久形态靠 1D grid-stride，非持久形态需要 2D tile grid，而宿主侧 `*_split_grid()` 是按持久形态算的 → 只改宏 = 拿 1D grid 当 2D 用、越界写（实测整份输出 NaN） |
| `SPLIT` | 改变 partial 输出路径（是否需要 partial buffer、由谁归约） |
| `PD` | split 模式直写 partial（少一次 smem staging 与一个 barrier） |
| `EPI` | 融合 epilogue 写 gbuf/raw1（仅在 `SPLIT=1` 下实现） |
| `RESID` | epilogue 追加残差项（改变写出的出口） |
| `APATH` | A 走 LDG→reg→STS（不同的发射路径与寄存器 staging） |
| `FA` | 级内 fp16 累加——**数值档改变**，生产尺度激活会溢出成 NaN |
| `F16P` | partial 用 fp16（同上，溢出风险） |
| `FAP` | `FA` 的促升周期（只在 `FA` 开时有意义） |

### 举例：看某相的当前值

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
#     ...（共 29 行，含 XS=2、PERSIST=1、RESID=1）
```

## 3. 子命令

| 子命令 | 干什么 | 关键参数（默认） |
| --- | --- | --- |
| `list` | 列出所有可扫几何；带 `--phase` 时打印它的轴名与当前值（位置序 = `--tiles` 序） | `--phase` |
| `capture` | 跑一次**真实推理**（拆相形态），用 `phase_check.Recorder` 把每个相的入口状态录成 case | `--model`（必填）、`--out`（必填）、`--kernels`、`--per-phase`（1）、`--cameras`、`--steps`、`--sampler`（euler）、`--seed`（0）、`--text-encoder-device`（**cuda**）、`--coop` |
| `run` | 扫一个候选集：位型对拍 + 交错 A/B | `<cases>`、`--phase`（必填）、`--axes "BN=32,48,64 BK=128"`、`--tiles`（完整 29 元组，可重复）、`--reps`（5）、`--iters`（100）、`--allow-drift`、`--allow-structural`、`--timeout`（600 s/候选） |
| `_one` | 内部：子进程量测**一个**候选并打一行 JSON（正常不用手敲） | `<cases>`、`--phase`、`--values`、`--src-dir`（省略 = 量生产扩展本身 = 基线行） |
| `apply` | 把选中的几何写回生产源码的宏（原地只改数字） | `--phase`、`--tiles`、`--geom`、`--dry-run` |
| `prune` | 删掉扫参 scratch 与所有变体扩展构建 | — |

## 4. 判据（两个都要满足才算可用）

**① 位型（bit pattern）**——对录制时的出口状态逐 buffer 比 `torch.equal`。改 blocking 参数
（`BN`/`BK`/stage 数/warp 划分）时 k 的累加顺序不变，**应当逐位不变**；报出差异只有两种可能：该轴
天然改数值（`SPLIT`/`FA`/`F16P`/`FAP`/`PERSIST`/`PD`/`EPI`/`RESID`），或者算错了。两者都不该被
「静默采纳」，所以默认只推荐逐位一致的候选，想改数值必须显式 `--allow-drift`。

**② 有限性（finiteness）**——输出里任何 inf/nan 当场作废：`FA=1`（级内 fp16 累加）在小激活上跑得
好好的，到生产尺度激活会变成整片 NaN。

**计时口径**（报告里三列）：

- `candidate` 列只列**与原值不同的轴**（`= current geometry` 表示就是当前几何，即基线）；
- `bit pattern` 列是判定；`min us` 是该实现自己的绝对耗时；`rel` 是与**同进程内重测的基线**的比值
  —— 跨进程的绝对值会漂几个百分点，**只有 `rel` 可比**；
- 每行末尾若出现 `<- not adopted`，说明它不可用（编译失败 / 崩溃 / 非有限 / 改数值且未开
  `--allow-drift`）。

两条自检：**基线永远是第一行**（现几何 vs 现几何，`rel` 应当 ≈ `1.000`，这是计时方法本身的自检）；
若某个子进程里**基线自己**都对不上录制出口（`prod_ok` 为假），脚本会警告「本轮数字不可信」——先查这
个相是不是确定性的（registry 里 `reduction` 应当是 fixed）。

## 5. 一次完整扫参

```bash
cd TyBoK

# 0) 录一次真实推理的相位入口（拆相形态；一次录制覆盖全部 41 个相）
python -m tybok.policies.fastwam.kernels.sweep capture \
    --model /path/to/fastwam_checkpoint --out /tmp/cases.pt

# 1) 看这个相有哪些轴、当前值是多少
python -m tybok.policies.fastwam.kernels.sweep list --phase vdit.ffn/P4

# 2) 扫：只给要动的轴（其余沿用当前宏值），候选 = 当前几何 + 这些轴的笛卡尔积
python -m tybok.policies.fastwam.kernels.sweep run /tmp/cases.pt \
    --phase vdit.ffn/P4 --axes "BN=32,48,64 BK=64,128"
#    也可以直接给完整 29 元组（可重复）：--tiles "64 48 128 3 3 4 2 1 ..."

# 3) 采纳 run 末尾打印的 apply 命令（先 --dry-run 看一眼 diff）
python -m tybok.policies.fastwam.kernels.sweep apply --phase vdit.ffn/P4 \
    --tiles "…29 个值…" --dry-run

# 4) 写回后（顺序不要颠倒）：重编译 → 对账 → 回归
python -m tybok.policies.fastwam.kernels.geometry          # 对账：报哪一项与二进制不一致
python -m tybok.policies.fastwam.kernels.geometry --bootstrap   # 只在需要重建表时
python tests/run_tests.py --models fastwam                  # 三族逐位门（见 tests/README.zh-CN.md）

# 5) 回收扫参产生的 scratch 与变体扩展
python -m tybok.policies.fastwam.kernels.sweep prune
```

`run` 只在最后打印「最快的可用候选」；如果它就是当前几何，会直接说「nothing better, no need to touch
the macro」。

## 6. 设计上的三条硬规矩

1. **一个候选一个子进程**。坏几何不是温柔失败：越过分档 smem 预算后 kernel 继续跑并越界写
   （报 illegal memory access），此时**整个 CUDA context 已经死了**，同进程里剩下的候选全部作废。
   崩溃/超时只记一行（`invalid: ...`）然后继续下一个。
2. **取 min，不取中位数**；**不要在测量循环里录图**。逐个回放的抖动比要找的差值还大，`min` 才是
   微基准的稳健估计（对应「没被打扰的那一轮」）；capture 要几十毫秒且会扰动状态，所以每个实现先各录
   一张 CUDA graph，之后只轮流 replay，奇偶轮交换顺序。
3. **不改生产源码 + 按内容哈希命名**。`kernels/` 整份复制到 scratch（`$FASTWAM_SWEEP_DIR`，默认
   `~/.cache/fastwam_geom_sweep/<内容哈希>/`），只原地换目标宏的数字（逗号、空格、续行位置都不动）；
   宏没真正改到（内容哈希与生产相同）会当场报错——否则扫出来的数字全是假的；同一组几何只编一次。
   `smem` 估算只是提前告警（`row = BK + (16 if XS==2 else 0)`、`est = BM*row*SA + BN*row*SB`），
   kernel 自己是权威（估高了=越界崩，估低了=子进程也会抱到）。

## 7. 注意事项

- **只在拆相形态下可扫**：cooperative 形态是单次 launch、没有相位（`--coop` 只影响录制内容，扫不了）。
- **capture 默认 `--text-encoder-device cuda`**（与 worker 默认的 `cpu` 不同）：UMT5 留在 CPU 时
  文本融合被引擎拒掉，`tmt5` 的 9 个相一个都录不到；一次录制要覆盖全部 41 个相，所以这里默认放 GPU
  （embedding 表仍在 CPU）。
- **换卡/换架构要重跑并重新对账**：几何表按 `(major, minor)` 声明（如 `(8, 9)` = sm_89）；表里没有
  的架构会如实报「表里没有这个架构」，而不是编一个值出来。
- **冷编译成本**：每个新几何都要编一次变体扩展，扫大网格时代价明显；`prune` 后可从头再来。
- `--allow-drift` 的候选可能与生产数值档不同（drift 档），采纳前务必跑回归；`--allow-structural`
  只是「强行编译」，不代表结果正确。
