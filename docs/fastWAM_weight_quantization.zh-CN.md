# fastWAM 权重量化方案（fp8 W8A8 常驻）

[English](fastWAM_weight_quantization.md) | [简体中文](fastWAM_weight_quantization.zh-CN.md)

fastWAM 后端把**两个 DiT 专家**（video expert / action expert）与 **UMT5 文本编码器**的大矩阵
权重离线量化成 **fp8 e4m3**，推理时按 **per-row（输出通道）缩放**反量化参与 GEMM，激活则按
**per-token** 在线量化。量化只发生在矩阵乘上：**bias、归一化、调制表、卷积、embedding 表与所有
小矩阵保持 bf16**。

- 量化实现：`tybok/policies/fastwam/models/fp8_linear.py`（自包含，只依赖 torch / triton）；
- 预量化产物：`model.fp8.safetensors`（DiT 两个专家 + proprio，其中 610 个矩阵存为 fp8）与
  `text_encoder/model.safetensors`（UMT5，168 个矩阵存为 fp8）。目录里含 `*.scale_weight` 的
  `*.safetensors` 即 fp8 文件，加载时**只**加载它（自包含，见 §5）。

## 1. 配方

### 权重（离线，写进 checkpoint）

对每个被选中的 `nn.Linear`（权重 `W`，形状 `[out, in]`）：

```
scale[i] = clamp(max_j |W[i, j]| / 448, 1e-12)        # 每个输出通道一个 fp32 scale
W8[i, j] = RNE(W[i, j] / scale[i])                    # 最近偶舍入 -> torch.float8_e4m3fn
```

- `448` = e4m3 的最大可表示值；`1e-12` 下限防整行为零时除零；
- 舍入就是 torch 的 `float8_e4m3fn` 转换（RNE），没有 per-tensor 二次缩放、没有随机舍入；
- checkpoint 里存成两个键：`<模块路径>.weight`（fp8，`[out, in]`）+ `<模块路径>.scale_weight`
  （fp32，`[out]`）。同一份文件里其余参数仍是 bf16，因此单文件自包含。

误差就是纯 e4m3 舍入：逐行最大绝对误差不超过该行 amax 的半个 ULP。在最上面那个 binade
（值域 256–448，ULP = 32，half-ULP = 16 个 scale 步）即 `16 / 448 = 3.57%`。
以本 checkpoint 实测（`video.blocks.0.self_attn.q.weight` 等）：`max|W8*scale - W| / row_amax =
3.5714e-2`（正好等于 `16/448`），且 **100% 的 fp8 码等于 `RNE(W/scale)`**（用文件里的
`scale_weight` 重新量化，逐字节一致）。

### 激活（在线，每个请求）

`FP8Linear.forward`：输入 `[M, K]` → per-token 量化 → fp8×fp8 → fp32 累加 → 反量化 + bias → bf16 输出：

```
sa[i] = clamp(max_k |x[i, k]| / 448, 1e-12)           # 每个 token 一个 fp32 scale
a8    = RNE(x / sa)                                   # 软件位操作，逐位复刻 torch 的 fp8 cast
out   = (a8 · W8) * sa[:, None] * scale[None, :] + bias    # fp32 累加与反量化
```

- 累加是 **fp32**，输出统一转 **bf16**（kernel 有可选 residual 分支，在反量化后按 fp32 相加）；
- 输入允许 **bf16**（norm / modulation / FFN 上游）或 **fp32**：o / cross-o 这类上游是
  SDPA 的 fp32 输出，走 **fp32 直接量化档**（一次 fp32→fp8 RNE，不会先降成 bf16）；
- 输入是 fp8 会直接报错——说明上游已经量化过一次（双重量化接线错误）。

## 2. 选中规则

**`nn.Linear` 且 `in_features ≥ 256` 且 `out_features ≥ 256`**（两个方向都够大）。

- 只有一侧大的矩阵留在 bf16：例如 video head 的 `192×3072`（out < 256）；
- 非 `nn.Linear`（模型里的 `Conv3d` patch_embedding、`nn.Embedding` 查找表）不参与；
- 阈值是**闭区间**，本 checkpoint 里最小的量化矩阵是 `1024×256`（action 的 time_embedding）。

本 checkpoint 共 **610** 个（video 305 + action 305）DiT 矩阵 + **168** 个 UMT5 矩阵。

## 3. 量化清单（键名为本 checkpoint 实际键名）

### video expert（`model.mot.mixtures.video`，30 层）

| 模型原语 | 键名 | 形状 | 数量 |
| --- | --- | --- | --- |
| `WanVideoDiT` self-attention q/k/v/o 投影 | `model.mot.mixtures.video.blocks.{0..29}.self_attn.{q,k,v,o}.weight` | 3072×3072 | 120 |
| cross-attention q/k/v/o 投影 | `model.mot.mixtures.video.blocks.{0..29}.cross_attn.{q,k,v,o}.weight` | 3072×3072 | 120 |
| FFN 升维 / 降维 | `model.mot.mixtures.video.blocks.{0..29}.ffn.{0,2}.weight` | 14336×3072 / 3072×14336 | 60 |
| 时间 embedding MLP | `model.mot.mixtures.video.time_embedding.{0,2}.weight` | 3072×256 | 2 |
| 时间投影（调制参数） | `model.mot.mixtures.video.time_projection.1.weight` | 18432×3072 | 1 |
| 文本 embedding MLP | `model.mot.mixtures.video.text_embedding.{0,2}.weight` | 3072×4096 | 2 |
| | | **合计** | **305** |

### action expert（`model.mot.mixtures.action`，30 层）

同一套模块结构，键名把 `video` 换成 `action`，形状按 action 的 hidden（1024）缩小：

| 模型原语 | 键名 | 形状 | 数量 |
| --- | --- | --- | --- |
| `ActionDiT` self-attention q/k/v/o | `model.mot.mixtures.action.blocks.{0..29}.self_attn.{q,k,v,o}.weight` | q/k/v 3072×1024，o 1024×3072 | 120 |
| cross-attention q/k/v/o | `model.mot.mixtures.action.blocks.{0..29}.cross_attn.{q,k,v,o}.weight` | q/k/v 3072×1024，o 1024×3072 | 120 |
| FFN 升维 / 降维 | `model.mot.mixtures.action.blocks.{0..29}.ffn.{0,2}.weight` | 4096×1024 / 1024×4096 | 60 |
| 时间 embedding MLP | `model.mot.mixtures.action.time_embedding.{0,2}.weight` | 1024×256 | 2 |
| 时间投影（调制参数） | `model.mot.mixtures.action.time_projection.1.weight` | 6144×1024 | 1 |
| 文本 embedding MLP | `model.mot.mixtures.action.text_embedding.{0,2}.weight` | 1024×4096 | 2 |
| | | **合计** | **305** |

### UMT5 文本编码器（`text_encoder/model.safetensors`，24 层）

| 模型原语 | 键名 | 形状 | 数量 |
| --- | --- | --- | --- |
| `UMT5Attention` q/k/v/o 投影 | `encoder.block.{0..23}.layer.{0,1}.SelfAttention.{q,k,v,o}.weight` | 4096×4096 | 96 |
| gated FFN `wi_0` / `wi_1` / `wo` | `encoder.block.{0..23}.layer.{0,1}.DenseReluDense.{wi_0,wi_1,wo}.weight` | 10240×4096 / 10240×4096 / 4096×10240 | 72 |
| | | **合计** | **168** |

## 4. 不量化（保持 bf16）

- **所有 bias**：包括 fp8 矩阵自己的 `bias`（`...self_attn.q.bias` 等），反量化时以 bf16 相加；
- **归一化与调制**：`blocks.{i}.norm3.weight`（+ action 侧 `norm3.bias`）、`self_attn.norm_q.weight` /
  `norm_k.weight`、`cross_attn.norm_q.weight` / `norm_k.weight`、`blocks.{i}.modulation`（`[1,6,D]`）、
  `video.head.modulation`（`[1,2,3072]`）；
- **小矩阵 / 非 Linear**：`video.patch_embedding.weight`（Conv3d `3072×48×1×2×2`）、
  `video.head.head.weight`（192×3072）、`action.head.weight`（14×1024）、
  `action.action_encoder.weight`（1024×14）、`proprio_encoder.weight`（4096×14，proprio 编码器）；
- **UMT5**：`SelfAttention.relative_attention_bias.weight`（32×64）、每层两个
  `layer_norm.weight`（4096）、`encoder.final_layer_norm.weight`、`shared.weight`
  （token embedding 表 256384×4096，约 2 GB）。

## 5. 部署侧怎么用

- **文件发现与直载**：目录里任何含 `*.scale_weight` 键的 `*.safetensors` 被识别为 fp8 文件
  （`*processor*.safetensors` 除外），此时**只加载 fp8 文件**——它与 bf16 文件键集合完全相同
  （本 checkpoint：两边各 915 个 `.weight`，fp8 文件额外多 610 个 `.scale_weight`）；
- **没有 dequant→requant 往返**：加载前用 `fp8ify_structural(min_dim=256)` 把这些 `nn.Linear`
  换成空的 `FP8Linear` 壳，loader 把 `.weight`（fp8）与 `.scale_weight` 原样拷入；
- **开关**：`--video-fp8` / `--action-fp8` / `--text-fp8` 默认开，`--video-bf16` / `--action-bf16` /
  `--text-bf16` 回退到 bf16；
- **UMT5 额外前提**：必须放在 GPU 上（`--text-encoder-device cuda`）才会 fp8 常驻；默认 `cpu`
  时 UMT5 保持 bf16（embedding 表也留在 CPU）；
- **不会隐式重量化**：目录里没有 fp8 文件时，即使显式开了 `--video-fp8` / `--action-fp8`，
  专家也保持 bf16 并打一条提示；
- **显存收益**：DiT 权重 12 GB（bf16 文件）→ 5.7 GB（fp8 文件）；启动日志口径为 video 专家
  ~5 GB、action 专家 ~1 GB 显存（与参数量相符：video ~5.0B、action ~1.0B，bf16→fp8 各省一半）。
  UMT5 的 fp8 常驻同样把权重字节减半（需 `--text-encoder-device cuda`，其 embedding 表默认留在
  CPU），实际占用随部署方式变化，这里不给具体数字；
- **与融合核的关系**：fp8 常驻是融合核的前置条件——action/video 的 `--cu-fused-*` 要求对应专家
  fp8 常驻，`--cu-fused-text-encoder` 还要求 UMT5 fp8 + CUDA + sm_89+；`--pack-qkv` 在 fp8 档下
  走 fp8 打包变体；
- **精度档**：属于 **drift 档**——与 bf16 参考**不逐位一致**，差异只来自上述量化本身（逐矩阵
  相对偏差约 3.7% 量级）；在其上再开 CUDA 融合核会叠加另一层 drift。

## 6. 自查

列出某 checkpoint 里所有被量化的矩阵（键名 = 模块路径）：

```python
from safetensors import safe_open

with safe_open("model.fp8.safetensors", framework="pt", device="cpu") as f:
    scales = [k for k in f.keys() if k.endswith(".scale_weight")]
print(len(scales))                       # 610
print(scales[0])                         # model.mot.mixtures.action.blocks.0.cross_attn.k.scale_weight
```

核对某个矩阵确实按本方案量化（应与上面 §1 的公式逐字节一致）：

```python
import torch
from safetensors import safe_open

key = "model.mot.mixtures.video.blocks.0.self_attn.q.weight"
with safe_open("model.fp8.safetensors", framework="pt", device="cpu") as f:
    w8 = f.get_tensor(key)
    sc = f.get_tensor(key[:-7] + ".scale_weight").float()
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    w = f.get_tensor(key).float()

requant = (w / sc[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)   # RNE
print((requant.view(torch.uint8) == w8.view(torch.uint8)).float().mean())     # 1.0
print(((w8.float() * sc[:, None] - w).abs().max(dim=1).values
       / w.abs().amax(dim=1)).max().item())                                   # 3.5714e-2 = 16/448
```
