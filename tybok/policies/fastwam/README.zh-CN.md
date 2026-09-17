# `fastwam` 后端（FastWAM / Wan2.2-MoT）

[English](README.md) | [简体中文](README.zh-CN.md)

`model_type="fastwam"` 的部署推理后端：Wan2.2-MoT 家族的 FastWAM 策略（单帧图像 → Wan VAE 编码 → UMT5 文本编码 → **video expert** 在 timestep 0 跑一次首帧 latent 并缓存每层 post-rope k/v → **action expert** 跑 Euler 去噪得到动作块）。基于 PyTorch `nn.Module`、参考 lerobot 的实现，并在其上做了自定义 Triton/CUDA 算子优化。本目录 = 引擎 + 模型 + CUDA 核 + 前/后处理。

## 融合优化的实现

### fp8 W8A8 常驻权重（默认开）

`models/fp8_linear.py` 是自包含的 fp8 基座：权重按行 fp8 e4m3 + fp32 scale、激活 per-token 动态量化（`amax/448` + 软件 RNE）、fp32 累加、bf16 输出，Triton GEMM。610 个 video/action 矩阵 + 168 个 UMT5 矩阵（都是 in/out ≥ 256 的 2-D Linear）走这条路；bias / norm / modulation / Conv3d / head 保持 bf16。逐键清单与量化配方见 [`docs/fastWAM_weight_quantization.zh-CN.md`](../../../docs/fastWAM_weight_quantization.zh-CN.md)。`--video-fp8` / `--action-fp8` / `--text-fp8` 分别控制三部分（都有对应的 `--*-bf16` 回退）。

### 融合核的两种形态与降级

每个融合族都有两种**数值一致**的形态：

- **协作式**（`--cooperative-kernel`）：核内用 `grid.sync()` 分相，grid 大小按**当前设备**的 occupancy × SM 数计算。它不需要独占整卡，但要求**整个 grid 同时驻留**——同一张卡上跑着别的负载（或 MPS 分区）时可能起不来；
- **拆相**（默认）：同样的核按相位拆成普通 launch，没有 grid 同驻约束，小卡 / 共享分区都能跑，代价是 launch 数更多。

`--cu-fused-*` 是**偏好而不是要求**：跑不了就降级并打一条 WARNING 说明原因，链路是 `fused（协作式）→ fused（拆相）→ eager`；`--require-fused` 把任何降级变成 `NotImplementedError`（部署 / CI 需要确定性时用）。降级判定全部在**破坏性打包之前**完成：`pack_fused` / pack-qkv 会释放原始的 q/k/v（以及 wi_0/wi_1）权重，打包之后就回不到 eager 路径了。

共同前置：CUDA 设备 + sm_89+（fp8 e4m3 mma）+ 对应专家 fp8 常驻；目录里没有 fp8 文件时就是 bf16，不做隐式重量化。

## 支持的命令行参数（仅模型推理相关）

通用入口参数（三个后端都一样）：`--model`（checkpoint 目录）、`--model-type`（后端覆盖，一般不用）、`--device`（`auto`/`cuda`/`cpu`）。worker 通道与网关参数见下文「worker / 网关参数」。

### 融合优化参数

| 参数 | 用途 |
|---|---|
| `--pack-qkv` | 每个 attention 的 q/k/v 打包成一个 GEMM（self 3×、cross 2×；fp8 用行拼接实现）：同一次 attention 输入只量化一次、GEMM launch 合并，对 fp8 更友好。**融合档下打包就是融合核的输入形式**：`--cu-fused-adit` 需要 action 侧已打包（没给本参数时引擎自动补上并打日志；打包会释放原始 q/k/v，之后融合核就是唯一路径），`--cu-fused-vdit` 不依赖本参数（`FusedVideoRunner` 逐层自己拼行），`--cu-fused-text-encoder` 恒走 `pack_fused` |
| `--cu-fused-adit` | action denoise 的每个 block 从"逐算子 torch 链"换成 **3 个融合 CUDA 核**：`adit.attn_self`（norm1+AdaLN 调制+量化 → packed qkv fp8 GEMM → qk-norm/RoPE → 对 video KV 缓存 + 新 32 行的 bf16 flash → o fp8 GEMM + gate 残差）、`adit.attn_cross`（norm3 → cross-q fp8 GEMM → RMS + masked bf16 flash（复用 context k/v）→ cross-o fp8 GEMM + 残差）、`adit.ffn`（norm2+AdaLN → up fp8 GEMM + GELU-tanh → 量化 → down fp8 GEMM + gate 残差）。前置：action expert fp8 常驻 + 默认的 action context 缓存；drift 档（chunk 相对 RMS ~2%） |
| `--cu-fused-vdit` | video prefill 的每个 block **整块**换成融合 CUDA 核（attention 的 self/cross 与 FFN 三段都换，不是只处理 block 尾部）：`vdit.attn_self` / `vdit.attn_cross` 把 `modulate(norm1) → qkv 投影 + qk-norm + 3-D RoPE → SDPA → gate 残差`（cross 版为 affine norm3 + 由 context 现算 k/v、无 RoPE）各压成 1 个核（6 phase / 5 grid.sync）；`vdit.ffn` 把 `modulate(apply_norm2(x)) → up → GELU → down → gate 残差` 压成 1 个核（4 phase / 3 grid.sync）。前置：video expert fp8 常驻；drift 档（rel ≤3e-2） |
| `--cu-fused-text-encoder` | UMT5 的两个子层各压成 1 个核：`tmt5.attn`（RMSNorm + per-token fp8 量化 → packed qkv fp8 GEMM → 带 pos-bias/causal 的注意力 → o fp8 GEMM + 残差）、`tmt5.ffn`（RMSNorm + packed `wi_0\|wi_1` fp8 GEMM + 激活 → down fp8 GEMM + 残差）。前置：`--text-encoder-device cuda` + fp8 文件。**embedding 表仍可留在 CPU**：融合核只覆盖 attention/FFN 的 fp8 矩阵，embedding 是纯查表，`--text-emb-cpu`（默认）与它无关 |
| `--action-pre-fused` | action DiT 前置组件融合（`kernels/action_pre.cu`）：时间路径（sinusoidal → time_embedding → time_projection）压成 4 次小 launch；context 预计算（原本 30 层各一次 FP8Linear + norm_k，同一份 context 被重复量化 30 次）压成"文本 embedding + 1 次量化 + 1 个大堆叠 GEMM + 1 次 repack"。需 `--cu-fused-adit` |
| `--video-pre-fused` | video DiT 非 block 组件的**纯精确**优化（`models/fused_video_pre.py`，逐位一致）：RoPE 复频表与预算的 fp64 正弦表常驻显存，`grid_sizes` / `video_mask` / `mot_mask` 按 key 缓存，并去掉融合入口里对已是连续 bf16 的 KV 反复 `.to().contiguous()` |
| `--cooperative-kernel` | 上面三族改用**协作式 launch**（见上文「融合核的两种形态与降级」：grid 按当前设备 occupancy × SM 数算，不独占整卡，但要求整个 grid 同驻）；不加则是拆相形态。协作式起不来时按降级链回退到拆相并打 WARNING |
| `--require-fused` | 融合档跑不了时**报错**而不是逐级降级（`fused 协作式 → fused 拆相 → eager`），用于部署 / CI 需要确定性时 |

### 其它性能开关

| 参数 | 用途 |
|---|---|
| `--graph` | 把 **VAE 帧编码 + DiT 核心**（video prefill + action 去噪）捕获成 CUDA Graph（逐位一致）；overlap 打开时（默认）是**单张多流图**，回放只有一次 `graph.replay()` |
| `--compile` | `torch.compile` DiT 核心的两个热点区域（`mot.prefill_video_cache` / `denoise_step`）。把编译范围收窄到热点区域（而不是整核 `_denoise_core`）**显著缩短了编译时间**：去噪循环留在 Python 层，步体只编译一次。可与 `--graph` 叠加；**设 `--compile` 时 overlap 自动关闭**并打一条 INFO。改数值（只保证流程正确）。融合档不能与它叠加（含 `--cu-fused-text-encoder`）：自研融合档一律不与 `--compile` 叠用，文本编码器虽然不在这两个区域内（`encode_prompt` 在 eager 与 graph 两条路径上都跑在它们之前），也不开特例 |
| `--overlap` / `--no-overlap` | 把 step-0 融进 video prefill：`prefill_video_layer` × `action_layer` 在模型内两流交错（video prefill 第 i 层在主 stream 跑，同时 action 第 i 层在侧 stream 上等该层 KV 事件后跑），step-0 藏在 prefill 窗口里；`--no-overlap` 改成严格顺序。逐位一致（同一批核，只是发射顺序不同）。**默认开**，eager 与 graph 两条路径都生效；`--compile` 时自动关闭 |
| `--video-fp8` / `--video-bf16` | video expert 用 fp8 常驻（默认，省 ~5GB 显存）还是 bf16 |
| `--action-fp8` / `--action-bf16` | action expert 用 fp8 常驻（默认，省 ~1GB 显存）还是 bf16 |
| `--text-fp8` / `--text-bf16` | UMT5 文本编码器用 fp8 还是 bf16（GPU 档） |
| `--no-prompt-cache` | 关闭 task prompt 的单条 memo（`FastWAM.encode_prompt` 按 prompt 字符串缓存一份 `context/context_mask`，warmup 打上、换 task 覆盖）。默认命中缓存后重复 task 只需真跑一次 DiT；加这个参数则**每个请求都真跑一次 UMT5**，逐位一致，供 `--profile` / 基准测试不被缓存美化 |
| `--no-action-context-cache` | 每步重算 action 专家的 cross-attention context（文本 embedding + 每层 k/v），而不是每个 chunk 只算一次。逐位一致，A/B 计时用 |

### 采样与诊断

| 参数 | 用途 |
|---|---|
| `--steps N` | 覆盖推理步数（`config.num_inference_steps`，默认 10） |
| `--seed N` | 推理随机种子（动作噪声） |
| `--sampler euler` | 仅支持 `euler`；传 `heun` 引擎直接报错 |
| `--profile` | 启动时跑一次延迟剖析并打印到日志（wall 时间；相位分解是 smolvla 专属）。注意 warmup 会先把 prompt 缓存打上，所以报的是命中后的稳态；要测未缓存成本请配 `--no-prompt-cache` |

### 模型 checkpoint 相关的参数

fastWAM 的权重（~12GB 的 MoT DiT + proprio encoder）全在 checkpoint 里，`text_encoder/`（fp8 UMT5）、`vae/`（Wan2.2 VAE）、`tokenizer/` 也放在同一目录，`config.json` 的 `text_encoder_model_id` / `tokenizer_model_id` / `vae_model_id` 指向这些相对子目录，整个目录可以直接搬走。因此 UMT5 / VAE / tokenizer 平时**不需要**额外参数；只有把它们放在 checkpoint 之外（或想换成另一份）时才用下面这几个：

- `--tokenizer-dir DIR`：覆盖 UMT5 tokenizer 目录；
- `--text-encoder-dir DIR`：覆盖 UMT5 权重目录（fp8 或 bf16 分片 + `config.json`）；
- `--vae-dir DIR`：覆盖 Wan2.2 VAE 目录（diffusers `AutoencoderKLWan` 权重 + `config.json`；默认取文本编码器目录的兄弟 `vae/`）；
- `--text-encoder-device cpu|cuda`：UMT5 放哪。默认 `cpu`（bf16，省显存、占主机内存），此时**文本编码是每个新 task 的主要成本**（CPU 上真跑一次 UMT5 是秒级，命中 task memo 后只剩 DiT 的成本）；放 `cuda` 后默认 fp8 常驻（需 fp8 文件），每个新 task 的成本大幅下降，代价是显存；
- `--text-emb-cpu` / `--no-text-emb-cpu`：embedding 表（2GB bf16，纯查表、无矩阵乘）默认留在 CPU，`--no-text-emb-cpu` 才把它搬到 GPU（仅 `--text-encoder-device cuda` 时相关）。

### worker / 网关参数

worker 通道与网关参数（不属于模型推理参数）三个后端共用，见 [docs/usage.zh-CN.md](../../../docs/usage.zh-CN.md) 的「参数」章节。

## 部署与客户端示例

```bash
# 一条命令拉起 worker + gateway（默认 ws://0.0.0.0:8765/ws）
python -m tybok serve \
    --model /path/to/fastwam \
    --model-type fastwam --graph --text-encoder-device cuda

# 客户端（examples/client.py）：task 字符串 + proprio state（本 checkpoint 14 维）
python examples/client.py --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
    --image image=frame.jpg
```

## 相机（图片）参数：和 pi0.5 / smolvla 不同

- **槽位名是完整键名**。fastwam 的 `describe()["cameras"]` 返回 checkpoint 声明的完整 observation key（本 checkpoint 是 `observation.images.image`），而 pi0.5 / smolvla 返回去掉前缀的短名（`image` / `camera1`）。网关两种写法都收（`observation.images.image` 与 `image` 等价），worker 侧统一成 `observation.images.<slot>`。
- **一个槽位 = 一帧拼接图**。checkpoint 声明的 n 个图像槽位会被 preprocessor 横向拼成**一帧**：每路先缩到 `image_size[1] // n` 宽，再沿宽度 `cat`（本 checkpoint 只有 1 个槽位，所以 `image_size = 384x320` 就是模型输入）。**多路视角要么客户端先拼好再发**（都塞进同一个槽位），**要么在 checkpoint 里声明多个槽位**、再用 `--camera-alias` 把客户端键改名到槽位。
- **不做 padding**。fastwam 的 `pad_mode` 是 `stretch`（直接缩放），pi0.5 是 center、smolvla 是 top-left。
- `--camera-alias SRC=DST[,SRC=DST...]`：SRC 是客户端发来的键，DST 是 checkpoint 的 image key（本 checkpoint 只有 `image`）；只在帧里缺 DST 时改名，多个映射写在同一个逗号分隔的值里。

## 不属于本模型的参数

- `--graph-cameras N[,M]`：**不是 fastwam 的参数**（仅为 CLI 统一而接受，引擎会说明它是 no-op）：被捕获的图只认一帧，与相机数无关；启动日志会打印槽位列表与目标分辨率。
- `--no-expert-prefix-kv-cache` → 引擎显式 `NotImplementedError`（eager / graph 路径始终用 video prefill 的 KV 缓存）。上游 lerobot fastwam 同样无条件：`wan/modular.py` 的 `prefill_video_cache` 一次缓存各层 video K/V，`_forward_action_cached` 每个去噪步直接 concat 复用，没有开关。
- `--rtc*` → 不支持 RTC（worker 层报错）。
- pi05/smolvla 专属：`--tl-fused-vit`、`--tl-llm-flash-attn`、`--tl-llm-fused-attn`、`--tl-vit-oproj`、`--tl-fused-expert`、`--tl-fp8-llm-mlp`、`--tl-fp8-expert-mlp`、`--vit-mlp-dtype`、`--skip-empty-cams`、`--pad-free` → `NotImplementedError`。
