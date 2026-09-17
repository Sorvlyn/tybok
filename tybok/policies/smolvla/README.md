# `smolvla` backend (SmolVLA)

[English](README.md) | [简体中文](README.zh-CN.md)

Deployment inference backend for `model_type="smolvla"`: a SmolVLM2-500M backbone (SigLIP vision encoder + SmolLM2 text decoder) plus a flow-matching action expert, built on PyTorch `nn.Module`, following the lerobot implementation, with custom Triton operator optimizations on top. This directory = engine + model + pre/post-processing.

## Implementation of the fused optimizations

The fused tiers are independent of one another and composable; all of them are **the framework's own Triton operator replacements**: with these tiers off, the eager path is bit-exact unchanged, and once they are on you are in a **drift tier** (not bit-exact with the eager reference).

- **Can be combined with `--graph`**: the fused operators get captured into the graph;
- **Not combined with `--compile`**: the custom fused tiers are never combined with `--compile`; for that combination the engine raises `NotImplementedError` directly;
- Only requires a CUDA device (no fp8 / architecture threshold).

## Supported command-line arguments (model inference only)

Common entry-point arguments (the same for all three backends): `--model` (checkpoint directory), `--model-type` (backend override, generally not needed), `--device` (`auto`/`cuda`/`cpu`). Worker channel and gateway arguments are covered in "worker / gateway arguments" below.

### Fused optimization arguments

| Argument | Purpose |
|---|---|
| `--tl-fused-expert` | Compresses each self-attention/cross-attention of the denoising path into a **single Triton GQA kernel** (`triton_self_attn` / `triton_cross_attn`): input RMSNorm + q (plus k/v for self) projection + RoPE + suffix KV write + GQA `expand/reshape` + 2-D mask construction + fp32 softmax, all inside one kernel (flash-style online softmax, one query head per program); at the same time the post-attention RMSNorm is fused into the MLP gate/up (`triton_norm_gate_up`: the norm weights are folded into the dot input, `silu(gate)*up` is computed in-kernel, and only the `[M, intermediate]` activations are written; `down_proj` still goes through cuBLAS) |
| `--tl-llm-fused-attn` | Replaces the LLM prefill attention chain with a single kernel: `triton_prefill_layer` = input RMSNorm + q/k/v projection + RoPE + KV-cache write (`_gqa_prefill_qkv_kernel`) + GQA flash (`_gqa_prefill_kernel`); `o_proj` / residual / MLP stay eager |
| `--tl-vit-oproj` | The vision `out_proj` becomes a single Triton GEMM (`triton_vision_out_proj`, consuming SDPA's `[B,H,L,D]` layout directly and saving the `[B*L,E]` bf16 copy from `transpose(1,2).contiguous()` in every layer); q/k/v are concatenated into one cuBLAS GEMM (required by this path); the vision MLP is not fused |

### Other performance switches

| Argument | Purpose |
|---|---|
| `--graph` | Captures the **whole** `VLAFlowMatching.sample_actions` (vision → prefix embed → LLM prefill → denoising loop) into **a single graph** (**bit-exact**). When overlap is on (the default), the interleaving of step-0 and prefill is a **model-level** two-stream branch, which stream capture turns into graph edges, so there is no need to split the graph layer by layer |
| `--graph-cameras N[,M]` | Which camera counts to **pre-capture** at startup (the configured camera count; values above the configured camera count are discarded). The camera count is a "prefix-length shape bucket" (each camera image goes through the ViT separately; what changes is the sequence length, not the batch); `--graph-cameras 2,3` means both 2 and 3 cameras are warmed up, and any other count is lazily captured the first time it appears |
| `--compile` | Uses `torch.compile` to compile the **inference stages** (`embed_prefix` + `vlm_with_expert.forward` + `denoise_step`), narrowing the compilation scope to the hot spots and markedly shortening compile time (the denoising loop stays in Python, so the step body is compiled only once). It can be combined with `--graph`: in that case only **`denoise_step`** is compiled (inside the whole-model graph), prefix embed / LLM prefill stay eager (a compiled prefill would break stream capture), and overlap is turned off automatically while an INFO line is logged. It changes numerics (only the flow is guaranteed correct). **Before compiling, the expert prefix KV projection cache is temporarily disabled** (restored according to the setting after compile/warmup/graph capture, with an INFO line for both operations). Fused tiers cannot be combined with it (see above) |
| `--overlap` / `--no-overlap` | Folds step-0 into the LLM prefill (`prefill_layer` × `step0_layer` model-level two streams, with step 0 hidden inside the prefill window); `--no-overlap` switches to strict sequential order. Bit-exact. The engine only enables it by default on CUDA + `euler` + `--graph` with `--compile` unset (eager / Heun / CPU keep the strict sequential path: the benefit is platform-dependent and not extensively tested); the implementation itself is independent of graph, and both modes (with and without graph) are bit-exact with the sequential path |
| `--no-expert-prefix-kv-cache` | Disables **this framework's own** expert prefix KV projection cache (model primitives `VLAFlowMatching.cache_expert_prefix_kv` / `SmolVLMWithExpertModel._cached_expert_prefix_kv`): by default, after the prefix prefill the fixed prefix K/V go through each layer's expert cross-attn `k_proj`/`v_proj` only once (cached by prefix length + KV fill count); turning it off recomputes them every step — bit-exact, just slower. Compared with lerobot: upstream only caches the **VLM-side prefix K/V** (`SmolVLAConfig.use_cache`, default True, no CLI), and the expert's `k_proj`/`v_proj` are recomputed every step, i.e. upstream is always equivalent to the "off" setting of this argument; `use_cache` still takes effect according to the checkpoint config and is independent of this argument |

### Sampling and diagnostics

| Argument | Purpose |
|---|---|
| `--steps N` | Overrides the number of denoising steps (defaults to the checkpoint config, usually 10). Fewer than the default steps changes the output |
| `--seed N` | Gives the denoising noise its own RNG so that the action sequence is reproducible (defaults to the global torch RNG) |
| `--sampler euler\|heun` | Denoising sampler, both are supported; `euler` is the bit-exact reference implementation, `heun` is second-order (N steps = 2N velocity evaluations, not bit-exact with the reference) |
| `--profile` | Runs a latency profiling pass once at startup and prints it to the log: wall-clock time + phase breakdown (phase names = the primitive being timed: `prepare_images` / `embed_prefix` / `vlm_with_expert.forward` / `_denoise_loop` / `postprocessor`) |

### Model checkpoint related arguments

The checkpoint holds the complete weights (`model.safetensors`) and the normalization stats; the **VLM backbone directory** is specified by `vlm_model_name` in `config.json` and may be an absolute path or a path relative to the checkpoint — this checkpoint places `SmolVLM2-500M-Video-Instruct/` inside the checkpoint directory, and the engine reads `config.json` (per-layer hidden size, etc.) and the tokenizer from there. So in normal use a single `--model` argument is all you need.

### worker / gateway arguments

The worker channel and gateway arguments (which are not model inference arguments) are shared by all three backends; see the "Arguments" section of [docs/usage.md](../../../docs/usage.md).

## Deployment and client example

```bash
# One command brings up worker + gateway (default ws://0.0.0.0:8765/ws)
python -m tybok serve \
    --model /path/to/smolvla \
    --graph --graph-cameras 2,3

# Client (examples/client.py): cameras are camera1 / camera2 / camera3, state is 6-dimensional
python examples/client.py --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0 \
    --image camera1=frame1.jpg,camera3=frame3.jpg
```

## Camera (image) arguments

- The client camera names are the **short names with the prefix stripped**: `camera1` / `camera2` / `camera3` (this checkpoint declares 3 slots; fastwam conversely returns the full key name `observation.images.image`, see its README).
- Resizing is **512x512 + top-left padding** (`pad_mode=top-left`), with pixel normalization to `[-1, 1]`; pi0.5 uses centered padding, and fastwam stretches directly.
- **Missing slots are not filled in**: only when `empty_cameras > 0` are the slots filled with an all-`-1` empty image (this checkpoint is 0), so sending one fewer camera means a **shorter prefix** and one fewer ViT pass; `--graph` buckets by camera count for exactly this reason, and `--graph-cameras 2,3` can warm up both the 2-camera and 3-camera shapes.
- `--camera-alias SRC=DST[,SRC=DST...]`: SRC is the key sent by the client, DST is the checkpoint's image key (e.g. `camera2`); the rename only happens when DST is missing from the frame, and multiple mappings go in the same comma-separated value.

## Arguments that do not belong to this model

- pi05-only: `--tl-fused-vit`, `--tl-llm-flash-attn`, `--tl-fp8-llm-mlp`, `--tl-fp8-expert-mlp`, `--vit-mlp-dtype`, `--skip-empty-cams`, `--pad-free` → `NotImplementedError`.
- fastwam-only: `--pack-qkv`, `--cu-fused-*`, `--text-*`, `--video-*`, `--action-*`, `--no-action-context-cache`, `--no-prompt-cache`, `--require-fused`, `--action-pre-fused`, `--video-pre-fused` → `NotImplementedError`.
- `--rtc*` → the worker raises an error (only pi05 supports RTC).
- `--tokenizer-dir`: not a smolvla engine argument; it is not passed in after signature filtering.
