# fastWAM weight quantization scheme (fp8 W8A8 resident)

[English](fastWAM_weight_quantization.md) | [简体中文](fastWAM_weight_quantization.zh-CN.md)

The fastWAM backend quantizes the big-matrix weights of the **two DiT experts** (video expert / action expert) and of the **UMT5 text encoder** offline into **fp8 e4m3**; at inference they are dequantized with a **per-row (output channel) scale** to take part in the GEMM, while activations are quantized online **per-token**. Quantization only happens on matrix multiplies: **bias, norm, modulation tables, convolution, embedding tables and all small matrices stay bf16**.

- Quantization implementation: `tybok/policies/fastwam/models/fp8_linear.py` (self-contained, depends only on torch / triton);
- Pre-quantized artifacts: `model.fp8.safetensors` (the two DiT experts + proprio, of which 610 matrices are stored as fp8) and `text_encoder/model.safetensors` (UMT5, 168 matrices stored as fp8). Any `*.safetensors` in the directory that contains `*.scale_weight` is an fp8 file, and in that case **only** it is loaded (self-contained, see §5).

## 1. Recipe

### Weights (offline, written into the checkpoint)

For each selected `nn.Linear` (weight `W`, shape `[out, in]`):

```
scale[i] = clamp(max_j |W[i, j]| / 448, 1e-12)        # one fp32 scale per output channel
W8[i, j] = RNE(W[i, j] / scale[i])                    # round to nearest even -> torch.float8_e4m3fn
```

- `448` = the maximum representable value of e4m3; the `1e-12` lower bound prevents division by zero when a whole row is zero;
- The rounding is just torch's `float8_e4m3fn` conversion (RNE), with no per-tensor secondary scaling and no stochastic rounding;
- In the checkpoint it is stored as two keys: `<module path>.weight` (fp8, `[out, in]`) + `<module path>.scale_weight`
  (fp32, `[out]`). The remaining parameters in the same file are still bf16, so a single file is self-contained.

The error is simply pure e4m3 rounding: the row-wise maximum absolute error does not exceed half a ULP of that row's amax. In the topmost binade
(value range 256–448, ULP = 32, half-ULP = 16 scale steps) that is `16 / 448 = 3.57%`.
Measured on this checkpoint (`video.blocks.0.self_attn.q.weight` etc.): `max|W8*scale - W| / row_amax =
3.5714e-2` (exactly equal to `16/448`), and **100% of the fp8 codes equal `RNE(W/scale)`** (requantizing with the
`scale_weight` in the file is byte-exact).

### Activations (online, per request)

`FP8Linear.forward`: input `[M, K]` → per-token quantization → fp8×fp8 → fp32 accumulation → dequantization + bias → bf16 output:

```
sa[i] = clamp(max_k |x[i, k]| / 448, 1e-12)           # one fp32 scale per token
a8    = RNE(x / sa)                                   # software bit manipulation, replicating torch's fp8 cast bit-for-bit
out   = (a8 · W8) * sa[:, None] * scale[None, :] + bias    # fp32 accumulation and dequantization
```

- The accumulation is **fp32**, and the output is uniformly converted to **bf16** (the kernel has an optional residual branch, added in fp32 after dequantization);
- The input may be **bf16** (norm / modulation / FFN upstream) or **fp32**: upstreams such as o / cross-o are
  the fp32 output of SDPA and go through the **direct fp32 quantization tier** (a single fp32→fp8 RNE, without first being lowered to bf16);
- An fp8 input raises an error directly — it means the upstream has already quantized once (a wiring mistake causing double quantization).

## 2. Selection rules

**`nn.Linear` with `in_features ≥ 256` and `out_features ≥ 256`** (both directions are large enough).

- Matrices that are large on only one side stay bf16: for example the video head's `192×3072` (out < 256);
- Non-`nn.Linear` modules (the model's `Conv3d` patch_embedding, `nn.Embedding` lookup tables) do not take part;
- The threshold is a **closed interval**, and the smallest quantized matrix in this checkpoint is `1024×256` (the action time_embedding).

This checkpoint has **610** DiT matrices in total (video 305 + action 305) + **168** UMT5 matrices.

## 3. Quantization inventory (the key names are this checkpoint's actual key names)

### video expert (`model.mot.mixtures.video`, 30 layers)

| Model primitive | Key name | Shape | Count |
| --- | --- | --- | --- |
| `WanVideoDiT` self-attention q/k/v/o projections | `model.mot.mixtures.video.blocks.{0..29}.self_attn.{q,k,v,o}.weight` | 3072×3072 | 120 |
| cross-attention q/k/v/o projections | `model.mot.mixtures.video.blocks.{0..29}.cross_attn.{q,k,v,o}.weight` | 3072×3072 | 120 |
| FFN up / down projection | `model.mot.mixtures.video.blocks.{0..29}.ffn.{0,2}.weight` | 14336×3072 / 3072×14336 | 60 |
| Time embedding MLP | `model.mot.mixtures.video.time_embedding.{0,2}.weight` | 3072×256 | 2 |
| Time projection (modulation parameters) | `model.mot.mixtures.video.time_projection.1.weight` | 18432×3072 | 1 |
| Text embedding MLP | `model.mot.mixtures.video.text_embedding.{0,2}.weight` | 3072×4096 | 2 |
| | | **Total** | **305** |

### action expert (`model.mot.mixtures.action`, 30 layers)

The same module structure, with `video` replaced by `action` in the key names and the shapes scaled down to the action hidden size (1024):

| Model primitive | Key name | Shape | Count |
| --- | --- | --- | --- |
| `ActionDiT` self-attention q/k/v/o | `model.mot.mixtures.action.blocks.{0..29}.self_attn.{q,k,v,o}.weight` | q/k/v 3072×1024, o 1024×3072 | 120 |
| cross-attention q/k/v/o | `model.mot.mixtures.action.blocks.{0..29}.cross_attn.{q,k,v,o}.weight` | q/k/v 3072×1024, o 1024×3072 | 120 |
| FFN up / down projection | `model.mot.mixtures.action.blocks.{0..29}.ffn.{0,2}.weight` | 4096×1024 / 1024×4096 | 60 |
| Time embedding MLP | `model.mot.mixtures.action.time_embedding.{0,2}.weight` | 1024×256 | 2 |
| Time projection (modulation parameters) | `model.mot.mixtures.action.time_projection.1.weight` | 6144×1024 | 1 |
| Text embedding MLP | `model.mot.mixtures.action.text_embedding.{0,2}.weight` | 1024×4096 | 2 |
| | | **Total** | **305** |

### UMT5 text encoder (`text_encoder/model.safetensors`, 24 layers)

| Model primitive | Key name | Shape | Count |
| --- | --- | --- | --- |
| `UMT5Attention` q/k/v/o projections | `encoder.block.{0..23}.layer.{0,1}.SelfAttention.{q,k,v,o}.weight` | 4096×4096 | 96 |
| gated FFN `wi_0` / `wi_1` / `wo` | `encoder.block.{0..23}.layer.{0,1}.DenseReluDense.{wi_0,wi_1,wo}.weight` | 10240×4096 / 10240×4096 / 4096×10240 | 72 |
| | | **Total** | **168** |

## 4. Not quantized (stay bf16)

- **All bias**: including the fp8 matrices' own `bias` (`...self_attn.q.bias` etc.), added in bf16 at dequantization;
- **Norm and modulation**: `blocks.{i}.norm3.weight` (+ `norm3.bias` on the action side), `self_attn.norm_q.weight` /
  `norm_k.weight`, `cross_attn.norm_q.weight` / `norm_k.weight`, `blocks.{i}.modulation` (`[1,6,D]`),
  `video.head.modulation` (`[1,2,3072]`);
- **Small matrices / non-Linear**: `video.patch_embedding.weight` (Conv3d `3072×48×1×2×2`),
  `video.head.head.weight` (192×3072), `action.head.weight` (14×1024),
  `action.action_encoder.weight` (1024×14), `proprio_encoder.weight` (4096×14, the proprio encoder);
- **UMT5**: `SelfAttention.relative_attention_bias.weight` (32×64), each layer's two
  `layer_norm.weight` (4096), `encoder.final_layer_norm.weight`, `shared.weight`
  (the token embedding table, 256384×4096, about 2 GB).

## 5. How to use it on the deployment side

- **File discovery and direct load**: any `*.safetensors` in the directory that contains a `*.scale_weight` key is recognized as an fp8 file
  (`*processor*.safetensors` excepted), and in that case **only the fp8 file is loaded** — its
  key set is exactly the same as the bf16 file's (this checkpoint: 915 `.weight` on each side, with the fp8 file having an extra 610 `.scale_weight`);
- **No dequant→requant round trip**: before loading, `fp8ify_structural(min_dim=256)` swaps these `nn.Linear`
  for empty `FP8Linear` shells, and the loader copies `.weight` (fp8) and `.scale_weight` in as they are;
- **Switches**: `--video-fp8` / `--action-fp8` / `--text-fp8` are on by default, and `--video-bf16` / `--action-bf16` /
  `--text-bf16` fall back to bf16;
- **Extra prerequisite for UMT5**: it must be placed on the GPU (`--text-encoder-device cuda`) to become fp8 resident; with the default `cpu`
  UMT5 stays bf16 (and the embedding table also stays on the CPU);
- **No implicit re-quantization**: when there is no fp8 file in the directory, the experts stay bf16 and a notice is logged even if `--video-fp8` / `--action-fp8` are explicitly enabled;
- **GPU memory savings**: DiT weights 12 GB (bf16 file) → 5.7 GB (fp8 file); the startup log reports about
  ~5 GB of GPU memory for the video expert and ~1 GB for the action expert (consistent with the parameter counts: video ~5.0B, action ~1.0B, bf16→fp8 saving half of each).
  The fp8 residency of UMT5 likewise halves the weight bytes (it requires `--text-encoder-device cuda`, and its embedding table stays on the
  CPU by default); the actual footprint varies with the deployment method, so no specific number is given here;
- **Relation to the fused kernels**: fp8 residency is a prerequisite of the fused kernels — the action/video `--cu-fused-*` require the corresponding expert to be
  fp8 resident, and `--cu-fused-text-encoder` additionally requires UMT5 fp8 + CUDA + sm_89+; `--pack-qkv` in the fp8 tier
  goes through the fp8 packing variant;
- **Precision tier**: this is the **drift tier** — it is **not bit-exact** with the bf16 reference, and the difference comes only from the quantization itself (a per-matrix
  relative deviation on the order of about 3.7%); enabling the CUDA fused kernels on top of it stacks another layer of drift.

## 6. Self-check

List all quantized matrices in a checkpoint (key name = module path):

```python
from safetensors import safe_open

with safe_open("model.fp8.safetensors", framework="pt", device="cpu") as f:
    scales = [k for k in f.keys() if k.endswith(".scale_weight")]
print(len(scales))                       # 610
print(scales[0])                         # model.mot.mixtures.action.blocks.0.cross_attn.k.scale_weight
```

Verify that a given matrix is indeed quantized according to this scheme (it should be byte-exact with the formula in §1 above):

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
