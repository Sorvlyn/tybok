# `fastwam` backend (FastWAM / Wan2.2-MoT)

[English](README.md) | [简体中文](README.zh-CN.md)

The deployment inference backend for `model_type="fastwam"`: the FastWAM policy of the Wan2.2-MoT family (single-frame image → Wan VAE encode → UMT5 text encode → the **video expert** runs once at timestep 0 over the first-frame latent and caches each layer's post-rope k/v → the **action expert** runs Euler denoising to produce the action chunk). Built on PyTorch `nn.Module`, following the lerobot implementation, with custom Triton/CUDA kernel optimizations on top. This directory = engine + model + CUDA kernels + pre/post-processing.

## Fused optimization implementation

### fp8 W8A8 resident weights (on by default)

`models/fp8_linear.py` is a self-contained fp8 substrate: weights are row-wise fp8 e4m3 + fp32 scale, activations are per-token dynamically quantized (`amax/448` + software RNE), fp32 accumulation, bf16 output, Triton GEMM. 610 video/action matrices + 168 UMT5 matrices (all 2-D Linears with in/out ≥ 256) go through this path; bias / norm / modulation / Conv3d / head stay bf16. For the per-key inventory and the quantization recipe see [`docs/fastWAM_weight_quantization.md`](../../../docs/fastWAM_weight_quantization.md). `--video-fp8` / `--action-fp8` / `--text-fp8` control the three parts separately (each has a corresponding `--*-bf16` fallback).

### The two forms of the fused kernels and fallback

Each fused tier has two **numerically identical** forms:

- **cooperative** (`--cooperative-kernel`): the kernel uses `grid.sync()` to split into phases, and the grid size is computed from the **current device**'s occupancy × SM count. It does not need exclusive use of the whole GPU, but it requires **the whole grid resident at once** — it may fail to launch when another workload (or an MPS shared partition) is running on the same GPU;
- **split-phase** (default): the same kernel is split by phase into ordinary launches, with no grid co-residence constraint, so a small GPU / a shared partition can run it too, at the cost of more launches.

`--cu-fused-*` is a **preference rather than a requirement**: if it cannot run, it falls back and prints a WARNING explaining why, along the chain `fused (cooperative) → fused (split-phase) → eager`; `--require-fused` turns any fallback into a `NotImplementedError` (use it when deployment / CI needs determinism). All fallback decisions are made **before destructive packing**: `pack_fused` / pack-qkv releases the original q/k/v (and wi_0/wi_1) weights, and after packing there is no way back to the eager path.

Common prerequisites: a CUDA device + sm_89+ (fp8 e4m3 mma) + the corresponding expert resident in fp8; when there is no fp8 file in the directory it is bf16, and no implicit re-quantization is done.

## Supported command-line arguments (model inference only)

Common entry-point arguments (identical for the three backends): `--model` (checkpoint directory), `--model-type` (backend override, generally not needed), `--device` (`auto`/`cuda`/`cpu`). For worker channel and gateway arguments see "worker / gateway arguments" below.

### Fused optimization arguments

| Argument | Purpose |
|---|---|
| `--pack-qkv` | Pack each attention's q/k/v into one GEMM (self 3×, cross 2×; implemented with row concatenation for fp8): a given attention input is quantized only once and the GEMM launches are merged, which is friendlier to fp8. **In the fused tier, packing is the input form of the fused kernel**: `--cu-fused-adit` requires the action side to be already packed (when this argument is not given the engine adds it automatically and logs it; packing releases the original q/k/v, after which the fused kernel is the only path), `--cu-fused-vdit` does not depend on this argument (`FusedVideoRunner` does its own row concatenation layer by layer), `--cu-fused-text-encoder` always goes through `pack_fused` |
| `--cu-fused-adit` | Replaces each action denoise block's "per-operator torch chain" with **3 fused CUDA kernels**: `adit.attn_self` (norm1+AdaLN modulation+quantization → packed qkv fp8 GEMM → qk-norm/RoPE → bf16 flash against the video KV cache + the new 32 rows → o fp8 GEMM + gate residual), `adit.attn_cross` (norm3 → cross-q fp8 GEMM → RMS + masked bf16 flash (reusing the context k/v) → cross-o fp8 GEMM + residual), `adit.ffn` (norm2+AdaLN → up fp8 GEMM + GELU-tanh → quantization → down fp8 GEMM + gate residual). Prerequisites: action expert resident in fp8 + the default action context cache; drift tier (chunk relative RMS ~2%) |
| `--cu-fused-vdit` | Replaces each video prefill block **as a whole** with a fused CUDA kernel (all three segments — attention self/cross and FFN — are replaced, not just the block tail): `vdit.attn_self` / `vdit.attn_cross` each compress `modulate(norm1) → qkv projection + qk-norm + 3-D RoPE → SDPA → gate residual` (the cross version being affine norm3 + k/v computed on the fly from the context, with no RoPE) into 1 kernel (6 phase / 5 grid.sync); `vdit.ffn` compresses `modulate(apply_norm2(x)) → up → GELU → down → gate residual` into 1 kernel (4 phase / 3 grid.sync). Prerequisites: video expert resident in fp8; drift tier (rel ≤3e-2) |
| `--cu-fused-text-encoder` | Each of UMT5's two sublayers is compressed into 1 kernel: `tmt5.attn` (RMSNorm + per-token fp8 quantization → packed qkv fp8 GEMM → attention with pos-bias/causal → o fp8 GEMM + residual), `tmt5.ffn` (RMSNorm + packed `wi_0\|wi_1` fp8 GEMM + activation → down fp8 GEMM + residual). Prerequisites: `--text-encoder-device cuda` + fp8 files. **The embedding table can still stay on the CPU**: the fused kernels only cover the attention/FFN fp8 matrices, embedding is a pure lookup, and `--text-emb-cpu` (default) is unrelated to it |
| `--action-pre-fused` | Fusion of the action DiT's pre-components (`kernels/action_pre.cu`): the time path (sinusoidal → time_embedding → time_projection) is compressed into 4 small launches; context precomputation (originally one FP8Linear + norm_k per each of the 30 layers, with the same context re-quantized 30 times) is compressed into "text embedding + 1 quantization + 1 large stacked GEMM + 1 repack". Requires `--cu-fused-adit` |
| `--video-pre-fused` | **Purely exact** optimization of the video DiT's non-block components (`models/fused_video_pre.py`, bit-exact): the RoPE frequency table and the precomputed fp64 sine table are resident in GPU memory, `grid_sizes` / `video_mask` / `mot_mask` are cached by key, and the repeated `.to().contiguous()` on KV that is already contiguous bf16 inside the fused entry point is removed |
| `--cooperative-kernel` | The three tiers above switch to a **cooperative launch** (see "The two forms of the fused kernels and fallback" above: the grid is computed from the current device's occupancy × SM count, it does not take exclusive use of the whole GPU, but requires the whole grid resident at once); without it, the split-phase form is used. When a cooperative launch cannot start, it falls back to split-phase along the fallback chain and prints a WARNING |
| `--require-fused` | When the fused tier cannot run, **raise an error** instead of falling back step by step (`fused cooperative → fused split-phase → eager`); for deployment / CI that needs determinism |

### Other performance switches

| Argument | Purpose |
|---|---|
| `--graph` | Captures **VAE frame encoding + the DiT core** (video prefill + action denoising) into a CUDA Graph (bit-exact); when overlap is on (default) it is a **single multi-stream graph**, and replay is a single `graph.replay()` |
| `--compile` | `torch.compile` the two hot regions of the DiT core (`mot.prefill_video_cache` / `denoise_step`). Narrowing the compilation scope to the hot regions (instead of the whole core `_denoise_core`) **significantly shortened compilation time**: the denoising loop stays in Python, and the step body is compiled only once. Can be stacked with `--graph`; **setting `--compile` turns overlap off automatically** and prints an INFO. Changes numerics (only the flow is guaranteed correct). The fused tier cannot be stacked with it (including `--cu-fused-text-encoder`): the in-house fused tier is never stacked with `--compile`, and although the text encoder is not inside those two regions (`encode_prompt` runs before them on both the eager and the graph path), no exception is made |
| `--overlap` / `--no-overlap` | Folds step-0 into video prefill: `prefill_video_layer` × `action_layer` interleave across two streams inside the model (video prefill layer i runs on the main stream while action layer i runs on the side stream after waiting for that layer's KV event), so step-0 hides inside the prefill window; `--no-overlap` switches to strict ordering. Bit-exact (the same batch of kernels, only the launch order differs). **On by default**, effective on both the eager and graph paths; turned off automatically with `--compile` |
| `--video-fp8` / `--video-bf16` | Whether the video expert is resident in fp8 (default, saves ~5GB of VRAM) or bf16 |
| `--action-fp8` / `--action-bf16` | Whether the action expert is resident in fp8 (default, saves ~1GB of VRAM) or bf16 |
| `--text-fp8` / `--text-bf16` | Whether the UMT5 text encoder uses fp8 or bf16 (GPU tier) |
| `--no-prompt-cache` | Disables the single-entry memo for the task prompt (`FastWAM.encode_prompt` caches one copy of `context/context_mask` per prompt string, populated during warmup and overwritten when the task changes). By default, once the cache is hit a repeated task needs only one real DiT run; with this argument **every request really runs UMT5**, bit-exact, so that `--profile` / benchmarks are not prettified by the cache |
| `--no-action-context-cache` | Recomputes the action expert's cross-attention context (text embedding + per-layer k/v) every step, instead of computing it only once per chunk. Bit-exact, for A/B timing |

### Sampling and diagnostics

| Argument | Purpose |
|---|---|
| `--steps N` | Override the number of inference steps (`config.num_inference_steps`, default 10) |
| `--seed N` | Inference random seed (action noise) |
| `--sampler euler` | Only `euler` is supported; passing `heun` makes the engine raise an error |
| `--profile` | Runs a latency profile once at startup and prints it to the log (wall time; the phase breakdown is smolvla-specific). Note that warmup populates the prompt cache first, so what is reported is the steady state after a cache hit; to measure the uncached cost, pair it with `--no-prompt-cache` |

### Model checkpoint related arguments

fastWAM's weights (~12GB MoT DiT + proprio encoder) are all inside the checkpoint, and `text_encoder/` (fp8 UMT5), `vae/` (Wan2.2 VAE) and `tokenizer/` also live in the same directory; `config.json`'s `text_encoder_model_id` / `tokenizer_model_id` / `vae_model_id` point at these relative subdirectories, so the whole directory can be moved as is. Therefore UMT5 / VAE / tokenizer normally **need** no extra arguments; only when you place them outside the checkpoint (or want to swap in another copy) do you use the following:

- `--tokenizer-dir DIR`: override the UMT5 tokenizer directory;
- `--text-encoder-dir DIR`: override the UMT5 weights directory (fp8 or bf16 shards + `config.json`);
- `--vae-dir DIR`: override the Wan2.2 VAE directory (diffusers `AutoencoderKLWan` weights + `config.json`; by default it takes the sibling `vae/` of the text encoder directory);
- `--text-encoder-device cpu|cuda`: where UMT5 lives. Default `cpu` (bf16, saves VRAM, uses host memory), in which case **text encoding is the main cost of every new task** (really running UMT5 once on the CPU takes seconds; after a task memo hit only the DiT cost remains); on `cuda` it is resident in fp8 by default (needs fp8 files), and the cost of every new task drops substantially, at the price of VRAM;
- `--text-emb-cpu` / `--no-text-emb-cpu`: the embedding table (2GB bf16, pure lookup, no matrix multiply) stays on the CPU by default, and only `--no-text-emb-cpu` moves it to the GPU (relevant only with `--text-encoder-device cuda`).

### worker / gateway arguments

worker channel and gateway arguments (which are not model inference arguments) are shared by the three backends; see the "Arguments" section of [docs/usage.md](../../../docs/usage.md).

## Deployment and client example

```bash
# Launch worker + gateway with one command (default ws://0.0.0.0:8765/ws)
python -m tybok serve \
    --model /path/to/fastwam \
    --model-type fastwam --graph --text-encoder-device cuda

# Client (examples/client.py): task string + proprio state (14-dim for this checkpoint)
python examples/client.py --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
    --image image=frame.jpg
```

## Camera (image) arguments: different from pi0.5 / smolvla

- **The slot name is the full key name.** fastwam's `describe()["cameras"]` returns the full observation key declared by the checkpoint (this checkpoint is `observation.images.image`), whereas pi0.5 / smolvla return the short names with the prefix stripped (`image` / `camera1`). The gateway accepts both spellings (`observation.images.image` and `image` are equivalent), and the worker side normalizes them to `observation.images.<slot>`.
- **One slot = one stitched frame.** The n image slots declared by the checkpoint are stitched horizontally by the preprocessor into **one frame**: each view is first resized to `image_size[1] // n` wide, then `cat` along the width (this checkpoint has only 1 slot, so `image_size = 384x320` is exactly the model input). **For multiple views, either the client stitches them first and sends them** (all stuffed into the same slot), **or multiple slots are declared in the checkpoint** and `--camera-alias` renames the client keys to the slots.
- **No padding.** fastwam's `pad_mode` is `stretch` (plain resize), pi0.5 is center and smolvla is top-left.
- `--camera-alias SRC=DST[,SRC=DST...]`: SRC is the key sent by the client and DST is the checkpoint's image key (this checkpoint only has `image`); renaming happens only when the frame lacks DST, and multiple mappings go in the same comma-separated value.

## Arguments that do not belong to this model

- `--graph-cameras N[,M]`: **not a fastwam argument** (accepted only for CLI uniformity, and the engine explains that it is a no-op): a captured graph recognizes only one frame, regardless of the number of cameras; the startup log prints the slot list and the target resolution.
- `--no-expert-prefix-kv-cache` → the engine raises `NotImplementedError` explicitly (the eager / graph paths always use the video prefill KV cache). Upstream lerobot fastwam is equally unconditional: `wan/modular.py`'s `prefill_video_cache` caches each layer's video K/V once, and `_forward_action_cached` concats and reuses them directly at every denoising step, with no switch.
- `--rtc*` → RTC is not supported (the worker layer raises an error).
- pi05/smolvla-specific: `--tl-fused-vit`, `--tl-llm-flash-attn`, `--tl-llm-fused-attn`, `--tl-vit-oproj`, `--tl-fused-expert`, `--tl-fp8-llm-mlp`, `--tl-fp8-expert-mlp`, `--vit-mlp-dtype`, `--skip-empty-cams`, `--pad-free` → `NotImplementedError`.
