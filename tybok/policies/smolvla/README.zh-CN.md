# `smolvla` 后端（SmolVLA）

[English](README.md) | [简体中文](README.zh-CN.md)

`model_type="smolvla"` 的部署推理后端：SmolVLM2-500M 主干（SigLIP 视觉编码器 + SmolLM2 文本解码器）加流匹配动作专家，基于 PyTorch `nn.Module`、参考 lerobot 的实现，并在其上做了自定义 Triton 算子优化。本目录 = 引擎 + 模型 + 前/后处理。

## 融合优化的实现

融合档彼此独立、可叠加，都是**框架自己的 Triton 算子替换**：不开这些档时 eager 路径逐位不变，开了之后属于 **drift 档**（与 eager 参考不逐位一致）。

- **可与 `--graph` 叠加**：融合后的算子会被捕获进图里；
- **不与 `--compile` 叠用**：自研融合档一律不与 `--compile` 叠加，组合由引擎直接报 `NotImplementedError`；
- 只要求 CUDA 设备（没有 fp8 / 架构门槛）。

## 支持的命令行参数（仅模型推理相关）

通用入口参数（三个后端都一样）：`--model`（checkpoint 目录）、`--model-type`（后端覆盖，一般不用）、`--device`（`auto`/`cuda`/`cpu`）。worker 通道与网关参数见下文「worker / 网关参数」。

### 融合优化参数

| 参数 | 用途 |
|---|---|
| `--tl-fused-expert` | 去噪路径的自注意力/交叉注意力各压成**单个 Triton GQA kernel**（`triton_self_attn` / `triton_cross_attn`）：输入 RMSNorm + q（self 另含 k/v）投影 + RoPE + suffix KV 写入 + GQA `expand/reshape` + 2-D mask 构造 + fp32 softmax，全在一个核内（flash 式在线 softmax，一 program 一个 query head）；同时把 post-attention RMSNorm 融进 MLP 的 gate/up（`triton_norm_gate_up`：norm 权重折进 dot 输入、核内算 `silu(gate)*up`，只写 `[M, intermediate]` 激活，`down_proj` 仍走 cuBLAS） |
| `--tl-llm-fused-attn` | LLM prefill 的注意力链换成一个核：`triton_prefill_layer` = input RMSNorm + q/k/v 投影 + RoPE + KV-cache 写入（`_gqa_prefill_qkv_kernel`）+ GQA flash（`_gqa_prefill_kernel`）；`o_proj` / 残差 / MLP 保持 eager |
| `--tl-vit-oproj` | 视觉 `out_proj` 变成单个 Triton GEMM（`triton_vision_out_proj`，直接吃 SDPA 的 `[B,H,L,D]` 布局，省掉每层 `transpose(1,2).contiguous()` 的 `[B*L,E]` bf16 拷贝）；q/k/v 拼成一个 cuBLAS GEMM（该路径要求）；视觉 MLP 不融合 |

### 其它性能开关

| 参数 | 用途 |
|---|---|
| `--graph` | 把**整核** `VLAFlowMatching.sample_actions`（vision → 前缀 embed → LLM prefill → 去噪循环）捕获成**一张图**（**逐位一致**）。overlap 打开时（默认），step-0 与 prefill 的交错是**模型内**两流分支，被 stream capture 变成图的边，不需要逐层拆图 |
| `--graph-cameras N[,M]` | 启动时**预捕获**哪些相机数（默认配置的相机数；超过配置相机数的值会被丢弃）。相机数是"前缀长度形状桶"（每相机一幅图单独过一次 ViT，改变的是序列长度，不是 batch）；`--graph-cameras 2,3` 表示 2/3 相机都预热好，其它数量在首次出现时惰性捕获 |
| `--compile` | 用 `torch.compile` 编译**推理阶段**（`embed_prefix` + `vlm_with_expert.forward` + `denoise_step`），把编译范围收窄到热点、明显缩短编译时间（去噪循环留在 Python 层，所以步体只编译一次）。可与 `--graph` 叠加：此时只编译 **`denoise_step`**（在整核图内部），前缀 embed / LLM prefill 保持 eager（编译过的 prefill 会让流捕获失效），并自动关掉 overlap 同时打一条 INFO。改数值（只保证流程正确）。**编译前会临时关掉专家前缀 KV 投影缓存**（编译/预热/图捕获完成后按设置恢复，两次操作都打 INFO）。融合档不能与它叠加（见上） |
| `--overlap` / `--no-overlap` | 把 step-0 融进 LLM prefill（`prefill_layer` × `step0_layer` 模型级两流，step 0 藏在 prefill 窗口里）；`--no-overlap` 改成严格顺序。逐位一致。引擎只在 CUDA + `euler` + `--graph` + 未设 `--compile` 时默认打开它（eager / Heun / CPU 保持严格顺序路径：收益依平台、未广泛测试）；实现本身与 graph 无关，有/无 graph 两种模式都与顺序路径逐位一致 |
| `--no-expert-prefix-kv-cache` | 关闭**本框架自己的**专家前缀 KV 投影缓存（模型原语 `VLAFlowMatching.cache_expert_prefix_kv` / `SmolVLMWithExpertModel._cached_expert_prefix_kv`）：默认在前缀 prefill 之后只把固定的前缀 K/V 过一次每层专家 cross-attn 的 `k_proj`/`v_proj`（按前缀长度 + KV 填充计数缓存），关掉则每步重算——逐位一致，仅慢。对照 lerobot：上游只缓存 **VLM 侧前缀 K/V**（`SmolVLAConfig.use_cache`，默认 True、无 CLI），专家的 `k_proj`/`v_proj` 每步都重算，即上游恒等于本参数的"关"档；`use_cache` 仍按 checkpoint 配置生效，与本参数无关 |

### 采样与诊断

| 参数 | 用途 |
|---|---|
| `--steps N` | 覆盖去噪步数（默认取 checkpoint 配置，通常 10）。少于默认步数会改变输出 |
| `--seed N` | 给去噪噪声单独的 RNG，使动作序列可复现（默认用全局 torch RNG） |
| `--sampler euler\|heun` | 去噪采样器，两者都支持；`euler` 是逐位一致的参考实现，`heun` 为二阶（N 步 = 2N 次速度评估，不与参考逐位一致） |
| `--profile` | 启动时跑一次延迟剖析并打印到日志：wall 时间 + 阶段分解（阶段名 = 被计时的那条原语：`prepare_images` / `embed_prefix` / `vlm_with_expert.forward` / `_denoise_loop` / `postprocessor`） |

### 模型 checkpoint 相关的参数

checkpoint 里是完整权重（`model.safetensors`）和归一化 stats；**VLM 骨干目录**由 `config.json` 的 `vlm_model_name` 指定，支持绝对路径或相对 checkpoint 的路径——本 checkpoint 就把 `SmolVLM2-500M-Video-Instruct/` 放在 checkpoint 目录内，引擎从那里读 `config.json`（各层 hidden size 等）和 tokenizer。所以平时只有 `--model` 一个参数就够了。

### worker / 网关参数

worker 通道与网关参数（不属于模型推理参数）三个后端共用，见 [docs/usage.zh-CN.md](../../../docs/usage.zh-CN.md) 的「参数」章节。

## 部署与客户端示例

```bash
# 一条命令拉起 worker + gateway（默认 ws://0.0.0.0:8765/ws）
python -m tybok serve \
    --model /path/to/smolvla \
    --graph --graph-cameras 2,3

# 客户端（examples/client.py）：相机为 camera1 / camera2 / camera3，state 6 维
python examples/client.py --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0 \
    --image camera1=frame1.jpg,camera3=frame3.jpg
```

## 相机（图片）参数

- 客户端相机名是**去掉前缀的短名**：`camera1` / `camera2` / `camera3`（本 checkpoint 声明了 3 个槽位；fastwam 反过来返回完整键名 `observation.images.image`，见其 README）。
- 缩放是 **512x512 + top-left padding**（`pad_mode=top-left`），像素归一到 `[-1, 1]`；pi0.5 是居中 padding，fastwam 是直接拉伸。
- **缺的槽位不补齐**：只有 `empty_cameras > 0` 时才会用整幅 `-1` 的空图填满（本 checkpoint 是 0），所以少发一路相机就是**前缀更短**、少过一次 ViT；`--graph` 正是按相机数分桶的，`--graph-cameras 2,3` 可以把 2/3 相机两种形状都预热好。
- `--camera-alias SRC=DST[,SRC=DST...]`：SRC 是客户端发来的键，DST 是 checkpoint 的 image key（如 `camera2`）；只在帧里缺 DST 时改名，多个映射写在同一个逗号分隔的值里。

## 不属于本模型的参数

- pi05 专属：`--tl-fused-vit`、`--tl-llm-flash-attn`、`--tl-fp8-llm-mlp`、`--tl-fp8-expert-mlp`、`--vit-mlp-dtype`、`--skip-empty-cams`、`--pad-free` → `NotImplementedError`。
- fastwam 专属：`--pack-qkv`、`--cu-fused-*`、`--text-*`、`--video-*`、`--action-*`、`--no-action-context-cache`、`--no-prompt-cache`、`--require-fused`、`--action-pre-fused`、`--video-pre-fused` → `NotImplementedError`。
- `--rtc*` → worker 报错（仅 pi05 支持 RTC）。
- `--tokenizer-dir`：非 smolvla 引擎参数，签名过滤后不会传入。
