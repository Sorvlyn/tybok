# `pi05` 后端（pi0.5）

[English](README.md) | [简体中文](README.zh-CN.md)

`model_type="pi05"` 的部署推理后端：PaliGemma（SigLIP-SO400M 视觉塔 + Gemma-2B 语言模型）+ AdaRMS 动作专家 + 流匹配去噪，基于 PyTorch `nn.Module`、参考 lerobot 的实现，并在其上做了自定义 Triton 算子优化。本目录 = 引擎 + 模型 + 前/后处理；也是三个后端里唯一支持 Real-Time Chunking（RTC）的。

## 融合优化的实现

融合档彼此独立、可叠加，都是**框架自己的 Triton 算子替换**：不开这些档时 eager 路径逐位不变，开了之后属于 **drift 档**（与 eager 参考不逐位一致）。

- **可与 `--graph` 叠加**：融合后的算子会被捕获进图里；
- **不与 `--compile` 叠用**：自研融合档一律不与 `--compile` 叠加，组合由引擎直接报 `NotImplementedError`；
- **fp8 档需要 sm_89+**（fp8 e4m3 tensor-core MMA）：`--tl-fp8-llm-mlp` / `--tl-fp8-expert-mlp` 在不支持的卡上**加载权重之前**就报错；其余档只要求 CUDA 设备。

## 支持的命令行参数（仅模型推理相关）

通用入口参数（三个后端都一样）：`--model`（checkpoint 目录）、`--model-type`（后端覆盖，一般不用）、`--device`（`auto`/`cuda`/`cpu`）。worker 通道与网关参数见下文「worker / 网关参数」。

### 融合优化参数

| 参数 | 用途 |
|---|---|
| `--tl-fused-vit` | SigLIP 视觉塔（27 层、原本 fp32 走 FFMA）低精度融合：`triton_vision_qkv` 把 `layer_norm1 → q/k/v` 三次投影合成一个 fp16 tensor-core GEMM（LN 统计仍 fp32、输出写回 fp32，好让 SDPA/softmax 保持原样）；`triton_vision_mlp` 把 `layer_norm2 → fc1 → gelu_tanh → fc2` 融合成两个 bf16 GEMM；另外把 `post_layernorm + multi_modal_projector` 融成一个 bf16 Triton GEMM。残差加法仍走 torch（残差流保持 fp32） |
| `--tl-llm-flash-attn` | VLM prefill 的注意力：GQA 原生 flash（`triton_prefill_attn`）替掉 `expand→reshape` 的 GQA 物化 + bf16 SDPA；配套用 cuBLAS 做 q/k/v 拼接的单次 GEMM（`GemmaAttention._forward_fused_prefill`） |
| `--tl-fused-expert` | 去噪专家每层 3 个核：`triton_qkv_rope_fused`（AdaRMS 归一 + packed q/k/v 投影 + 寄存器内 RoPE，一个核完成）→ split-K GQA flash → MLP 侧 `triton_expert_mlp`（AdaRMS + gate/up + `gelu_tanh`，只写 `[M, intermediate]`，down 走 cuBLAS）+ `triton_expert_residual`（把 `residual + out*gate` 的三个小核压成一个、逐位同 bf16 双重舍入）；并把专家末端 norm + `action_out_proj` 一并融合（`fuse_final_tail` → `final_norm_out_proj`） |
| `--tl-fp8-llm-mlp` | VLM prefill MLP（2048→16384）走 W8A8 fp8：权重离线 per-channel 量化（转置后的 fp8 权重预算好，调用时不再转置）+ 激活 per-token 量化 + 单次 Triton GEMM 在 epilogue 里同时乘两个 scale 并直接写 bf16。需 sm_89+，否则启动即报错 |
| `--tl-fp8-expert-mlp` | 去噪专家 MLP 的 fp8 生产流水（3 个核，无主机侧激活往返）：`triton_norm_gate_up_fp8`（AdaRMS + fp8 gate/up + gelu，同时产出 bf16 激活与 per-CTA 的 `\|act\|` 部分最大值）→ `triton_quant_act_fp8`（归约出 per-token scale 并转 fp8）→ `triton_down_proj_fp8_splitk`（K 方向切分的 fp8 down GEMM + fp32 合并）。需 sm_89+ |
| `--vit-mlp-dtype {fp16,bf16}` | 不是 Triton 档：只把 SigLIP 塔 MLP（fc1/fc2）的 GEMM 换成 fp16/bf16（norms / attention / residual 仍 fp32），仅 eager ViT 生效；`--tl-fused-vit` 下被忽略（该档自己决定 MLP 精度） |

### 其它性能开关

| 参数 | 用途 |
|---|---|
| `--graph` | 把模型整核 `sample_actions`（vision → 前缀 embedding → LLM prefill → 去噪循环）捕获成**一张** CUDA Graph，请求侧 replay 一次；overlap 打开时（默认）step-0 专家层与 prefill 交错的那段并行分支也在同一张图里（模型内的副流 fork/join 被 stream capture 变成图的边）。逐位一致 |
| `--graph-cameras N[,M]` | 启动时预捕获的相机数（前缀长度形状桶；每相机一幅图单独过一次 ViT，改的是序列长度而非 batch）。**不带 `--pad-free` 时本参数不生效**（键固定为 checkpoint 的全部 image slot 数）；带 `--pad-free` 时默认 2、可用本参数覆盖（超过 checkpoint 真实相机数的值会被丢弃），其它数量首次出现时惰性捕获 |
| `--compile` | `torch.compile` 编译**推理阶段**（`embed_prefix` + `paligemma_with_expert.forward` + `denoise_step`），把编译范围收窄到热点、明显缩短编译时间（Euler 循环留在 Python 层，步体只编译一次）。可与 `--graph` 叠加：捕获的整核正是 `sample_actions`，这些编译区域在其中；叠加时自动关闭 overlap 并打一条 INFO。改数值（只保证流程正确）。融合档不能与它叠加（见上） |
| `--overlap` / `--no-overlap` | 把 step-0 融进 LLM prefill（`prefill_layer` × `step0_layer` 模型内两流交错，step 0 藏在 prefill 窗口里）；`--no-overlap` 改成严格顺序。逐位一致。引擎只在 `--graph` + CUDA + 未设 `--compile` 时默认打开它（eager 下收益依平台、未广泛测试）；实现本身与 graph 无关，有/无 graph 两种模式都与顺序路径逐位一致 |
| `--skip-empty-cams` | 空相机槽（如 checkpoint 里的 `empty_camera_*` 占位槽）**整块跳过 ViT + projector**（`embed_image` 不进），用同形状零向量顶上：位置/mask 不变，因此逐位一致。与融合档无关——`--tl-fused-vit` 下同样生效（跳过判断在 `embed_prefix` 里、发生在进 ViT 之前）。**不适用于 CUDA-graph 捕获路径**：该判断需要一次 `img_mask.all()` 的 D2H 同步，图内非法（`use_static` 下不进入该分支），启动时会打一条 INFO 说明；graph 模式请用 `--pad-free` |
| `--pad-free` | padding-free VLM prefill：丢掉 `empty_camera_*` 占位槽，并把语言打包到 16 的倍数桶。eager 与 graph 用同一套规则，因此 graph 仍逐位一致 |

### 采样与诊断

| 参数 | 用途 |
|---|---|
| `--steps N` | 覆盖去噪步数（默认取 checkpoint 配置，通常 10） |
| `--seed N` | 去噪噪声独立 RNG（可复现；默认全局 torch RNG） |
| `--sampler euler` | 仅支持 `euler`（逐位一致）；传 `heun` 引擎直接报错 |
| `--profile` | 启动时跑一次延迟剖析并打印到日志（wall 时间；相位分解是 smolvla 专属） |

### Real-Time Chunking（本后端独有）

| 参数 | 用途 |
|---|---|
| `--rtc` | 打开 RTC 引导（纯推理数学，无额外权重）。之后由客户端按请求驱动：`prev_chunk_left_over` / `inference_delay` / `execution_horizon`，并额外收到归一化后的 chunk |
| `--rtc-schedule linear\|zeros\|ones\|exp` | 前缀注意力权重调度的形状（默认 `linear`） |
| `--rtc-max-guidance-weight F` | RTC 引导权重上限（默认 10.0） |
| `--rtc-execution-horizon N` | 执行视界步数（默认 10，可按请求覆盖） |
| `--rtc-debug` | 记录每步 RTC 引导的调试信息（Tracker；默认关） |
| `--rtc-debug-maxlen N` | 调试 Tracker 的滑窗长度（默认 100） |

RTC 与 `--graph` 互斥（RTC 引导在 `enable_grad` 下跑，图外执行 autograd 会破坏捕获的内存池）；另外只有 pi05 支持 RTC，其它后端传 `--rtc*` 会被 worker 拒绝。

### 模型 checkpoint 相关的参数

pi05 的权重全在 checkpoint 内（`model.safetensors` + `policy_preprocessor`/`policy_postprocessor` 的归一化 stats）；tokenizer 目录从 `policy_preprocessor.json` 的 `tokenizer_processor.tokenizer_name` 读取，支持绝对路径或相对 checkpoint 的路径。所以平时只有 `--model` 一个参数就够了，只有想换成另一份 tokenizer 时才用：

- `--tokenizer-dir DIR`：覆盖 PaliGemma tokenizer 目录（默认从 checkpoint 解析）。

### worker / 网关参数

worker 通道与网关参数（不属于模型推理参数）三个后端共用，见 [docs/usage.zh-CN.md](../../../docs/usage.zh-CN.md) 的「参数」章节。

## 部署与客户端示例

```bash
# 一条命令拉起 worker + gateway（默认 ws://0.0.0.0:8765/ws）
python -m tybok serve \
    --model /path/to/pi05 \
    --model-type pi05 --graph --profile

# 客户端（examples/client.py）：相机为 image / image2，state 8 维
python examples/client.py --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0,0,0 \
    --image image=frame.jpg,image2=wrist.jpg
```

## 相机（图片）参数

- 客户端相机名是**去掉前缀的短名**：`image` / `image2`（`describe()` 用的是 checkpoint 的**真实相机**列表，`empty_camera_0` 这类占位槽不会暴露给客户端；fastwam 反过来返回完整键名 `observation.images.image`，见其 README）。
- **缺的槽位会被补齐**：missing 的 image feature 由引擎补成整幅 `-1` 的图 + mask 0，前缀长度不变，所以只发一路相机也能跑（响应里的 `missing_cameras` 会列出缺的槽位）。
- 缩放是 **224x224 居中 padding**（`pad_mode=center`）：保持长宽比、居中填充；smolvla 是 top-left padding，fastwam 是直接拉伸。
- 空槽位的两种优化：`--skip-empty-cams`（eager 路径整块跳过 ViT，逐位一致）与 `--pad-free`（把占位槽从前缀里去掉，eager / graph 都能用）。
- `--camera-alias SRC=DST[,SRC=DST...]`：SRC 是客户端发来的键，DST 是 checkpoint 的 image key（如 `image2`）；只在帧里缺 DST 时改名，多个映射写在同一个逗号分隔的值里。
- `state` 会与 `task` 一起拼进文本 prompt（`Task: ..., State: <离散化状态>;\nAction: `），客户端照常发 state 数组即可。

## 不属于本模型的参数

- `--no-expert-prefix-kv-cache`：**不是 pi05 的参数**（引擎接受但不用它，仅为 CLI 统一）：上游 lerobot 的 pi05 没有这个开关（`configuration_pi05.py` 里无 `use_cache`），前缀 KV 复用是**结构性**的——prefill 用 `use_cache=True` 建好前缀 KV 缓存，每个去噪步都带着它、以 `use_cache=False`（只读不写，只算 suffix）复用它，所以没有可关的东西。
- smolvla 专属：`--tl-llm-fused-attn`、`--tl-vit-oproj` → `NotImplementedError`。
- fastwam 专属：`--pack-qkv`、`--no-action-context-cache`、`--no-prompt-cache`、`--cu-fused-*`、`--require-fused`、`--action-pre-fused`、`--video-pre-fused` → `NotImplementedError`。
- fastwam sidecar：`--text-encoder-dir`、`--vae-dir`、`--text-encoder-device`（非 `cpu` 时）、`--text-emb-cpu`、`--video-*`、`--action-fp8`、`--text-fp8` → `NotImplementedError`。
