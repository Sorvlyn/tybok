# `pi05` backend (pi0.5)

[English](README.md) | [简体中文](README.zh-CN.md)

The deployment inference backend for `model_type="pi05"`: PaliGemma (SigLIP-SO400M vision tower + Gemma-2B language model) + AdaRMS action expert + flow-matching denoising, built on PyTorch `nn.Module`, following the lerobot implementation, with custom Triton kernel optimizations on top. This directory = engine + model + pre/post-processing; it is also the only one of the three backends that supports Real-Time Chunking (RTC).

## Fused optimization implementation

The fused tiers are independent of each other and stackable, and are all **the framework's own Triton kernel replacements**: with these tiers off the eager path stays bit-for-bit identical, and once enabled they belong to the **drift tier** (not bit-for-bit identical to the eager reference).

- **Can be stacked with `--graph`**: the fused kernels are captured into the graph;
- **Not stacked with `--compile`**: the in-house fused tiers are never stacked with `--compile`, and the combination makes the engine raise `NotImplementedError` directly;
- **The fp8 tiers require sm_89+** (fp8 e4m3 tensor-core MMA): on unsupported cards `--tl-fp8-llm-mlp` / `--tl-fp8-expert-mlp` raise an error **before the weights are loaded**; the remaining tiers only require a CUDA device.

## Supported command-line arguments (model inference only)

Common entry-point arguments (identical for the three backends): `--model` (checkpoint directory), `--model-type` (backend override, generally not needed), `--device` (`auto`/`cuda`/`cpu`). For worker channel and gateway arguments see "worker / gateway arguments" below.

### Fused optimization arguments

| Argument | Purpose |
|---|---|
| `--tl-fused-vit` | Low-precision fusion of the SigLIP vision tower (27 layers, originally fp32 through FFMA): `triton_vision_qkv` merges the three `layer_norm1 → q/k/v` projections into one fp16 tensor-core GEMM (the LN statistics stay fp32 and the output is written back as fp32, so that SDPA/softmax keep their original behavior); `triton_vision_mlp` fuses `layer_norm2 → fc1 → gelu_tanh → fc2` into two bf16 GEMMs; it also fuses `post_layernorm + multi_modal_projector` into one bf16 Triton GEMM. The residual add still goes through torch (the residual stream stays fp32) |
| `--tl-llm-flash-attn` | Attention for the VLM prefill: native GQA flash (`triton_prefill_attn`) replaces the `expand→reshape` GQA materialization + bf16 SDPA; paired with a single cuBLAS GEMM for the concatenated q/k/v (`GemmaAttention._forward_fused_prefill`) |
| `--tl-fused-expert` | 3 kernels per denoising expert layer: `triton_qkv_rope_fused` (AdaRMS normalization + packed q/k/v projection + in-register RoPE, all done in one kernel) → split-K GQA flash → `triton_expert_mlp` on the MLP side (AdaRMS + gate/up + `gelu_tanh`, writing only `[M, intermediate]`, with down going through cuBLAS) + `triton_expert_residual` (compresses the three small kernels of `residual + out*gate` into one, bit-for-bit the same as bf16 double rounding); it also fuses the expert's trailing norm + `action_out_proj` (`fuse_final_tail` → `final_norm_out_proj`) |
| `--tl-fp8-llm-mlp` | The VLM prefill MLP (2048→16384) goes W8A8 fp8: offline per-channel weight quantization (the transposed fp8 weights are precomputed, so there is no transpose at call time) + per-token activation quantization + a single Triton GEMM that multiplies both scales in the epilogue and writes bf16 directly. Requires sm_89+, otherwise it raises an error at startup |
| `--tl-fp8-expert-mlp` | The fp8 production pipeline of the denoising expert MLP (3 kernels, no host-side activation round trips): `triton_norm_gate_up_fp8` (AdaRMS + fp8 gate/up + gelu, producing both the bf16 activations and the per-CTA partial max of `\|act\|`) → `triton_quant_act_fp8` (reduces to the per-token scale and converts to fp8) → `triton_down_proj_fp8_splitk` (an fp8 down GEMM split along K + fp32 merge). Requires sm_89+ |
| `--vit-mlp-dtype {fp16,bf16}` | Not a Triton tier: it only switches the GEMMs of the SigLIP tower MLP (fc1/fc2) to fp16/bf16 (norms / attention / residual stay fp32), in effect only for the eager ViT; ignored under `--tl-fused-vit` (that tier decides the MLP precision itself) |

### Other performance switches

| Argument | Purpose |
|---|---|
| `--graph` | Captures the model's whole-kernel `sample_actions` (vision → prefix embedding → LLM prefill → denoising loop) into **one** CUDA Graph, replayed once per request; when overlap is on (the default) the parallel branch where the step-0 expert layers interleave with the prefill is in the same graph too (the intra-model side-stream fork/join becomes graph edges under stream capture). Bit-for-bit identical |
| `--graph-cameras N[,M]` | The number of cameras pre-captured at startup (prefix-length shape buckets; each camera's image goes through the ViT separately, which changes the sequence length rather than the batch). **This argument has no effect without `--pad-free`** (the key is fixed to the checkpoint's total number of image slots); with `--pad-free` it defaults to 2 and can be overridden by this argument (values above the checkpoint's real camera count are dropped), and any other count is lazily captured the first time it appears |
| `--compile` | `torch.compile` compiles the **inference phases** (`embed_prefix` + `paligemma_with_expert.forward` + `denoise_step`), narrowing the compilation scope to the hot spots and markedly shortening compile time (the Euler loop stays in Python, so the step body is compiled only once). Can be stacked with `--graph`: the captured whole kernel is exactly `sample_actions`, and these compiled regions are inside it; stacking turns overlap off automatically and prints an INFO. Changes numerics (only the flow is guaranteed correct). The fused tiers cannot be stacked with it (see above) |
| `--overlap` / `--no-overlap` | Folds step-0 into the LLM prefill (`prefill_layer` × `step0_layer` interleave across two streams inside the model, with step 0 hidden in the prefill window); `--no-overlap` switches to strict ordering. Bit-for-bit identical. The engine only turns it on by default with `--graph` + CUDA and no `--compile` (under eager the benefit is platform-dependent and not widely tested); the implementation itself is independent of the graph, and both the graph and the no-graph modes stay bit-for-bit identical to the sequential path |
| `--skip-empty-cams` | Empty camera slots (such as the `empty_camera_*` placeholder slots in the checkpoint) **skip the ViT + projector entirely** (`embed_image` is not entered) and are topped up with a zero vector of the same shape: positions/mask are unchanged, hence bit-for-bit identical. Unrelated to the fused tiers — it also takes effect under `--tl-fused-vit` (the skip test is inside `embed_prefix` and happens before entering the ViT). **Not applicable to the CUDA-graph capture path**: that test needs one `img_mask.all()` D2H sync, which is illegal inside a graph (under `use_static` the branch is not entered), and one INFO is printed at startup to say so; use `--pad-free` in graph mode |
| `--pad-free` | padding-free VLM prefill: drops the `empty_camera_*` placeholder slots and packs the language into buckets that are multiples of 16. eager and graph use the same set of rules, so the graph stays bit-for-bit identical |

### Sampling and diagnostics

| Argument | Purpose |
|---|---|
| `--steps N` | Override the number of denoising steps (defaults to the checkpoint config, usually 10) |
| `--seed N` | Separate RNG for the denoising noise (reproducible; defaults to the global torch RNG) |
| `--sampler euler` | Only `euler` is supported (bit-for-bit identical); passing `heun` makes the engine raise an error directly |
| `--profile` | Runs a latency profile once at startup and prints it to the log (wall time; the phase breakdown is smolvla-specific) |

### Real-Time Chunking (unique to this backend)

| Argument | Purpose |
|---|---|
| `--rtc` | Turns on RTC guidance (pure inference-time math, no extra weights). It is then driven per request by the client: `prev_chunk_left_over` / `inference_delay` / `execution_horizon`, and the normalized chunk is returned in addition |
| `--rtc-schedule linear\|zeros\|ones\|exp` | The shape of the prefix attention weight schedule (default `linear`) |
| `--rtc-max-guidance-weight F` | Upper bound on the RTC guidance weight (default 10.0) |
| `--rtc-execution-horizon N` | The execution horizon in steps (default 10, overridable per request) |
| `--rtc-debug` | Records per-step RTC guidance debug info (Tracker; off by default) |
| `--rtc-debug-maxlen N` | Sliding window length of the debug Tracker (default 100) |

RTC and `--graph` are mutually exclusive (RTC guidance runs under `enable_grad`, and executing autograd outside the graph breaks the captured memory pool); also, only pi05 supports RTC, and `--rtc*` passed to the other backends is rejected by the worker.

### Model checkpoint related arguments

All of pi05's weights live inside the checkpoint (`model.safetensors` + the normalization stats of `policy_preprocessor`/`policy_postprocessor`); the tokenizer directory is read from `tokenizer_processor.tokenizer_name` in `policy_preprocessor.json`, and supports an absolute path or a path relative to the checkpoint. So normally `--model` alone is enough, and the following is only needed when you want to swap in another tokenizer:

- `--tokenizer-dir DIR`: override the PaliGemma tokenizer directory (parsed from the checkpoint by default).

### worker / gateway arguments

worker channel and gateway arguments (which are not model inference arguments) are shared by the three backends; see the "Arguments" section of [docs/usage.md](../../../docs/usage.md).

## Deployment and client example

```bash
# one command brings up worker + gateway (default ws://0.0.0.0:8765/ws)
python -m tybok serve \
    --model /path/to/pi05 \
    --model-type pi05 --graph --profile

# client (examples/client.py): cameras are image / image2, state is 8-dimensional
python examples/client.py --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0,0,0 \
    --image image=frame.jpg,image2=wrist.jpg
```

## Camera (image) arguments

- The client-side camera names are the **short names with the prefix stripped**: `image` / `image2` (`describe()` uses the checkpoint's list of **real cameras**; placeholder slots such as `empty_camera_0` are not exposed to the client, whereas fastwam is the other way round and returns the full key name `observation.images.image`, see its README).
- **Missing slots are backfilled**: a missing image feature is backfilled by the engine into an all-`-1` image + mask 0, and the prefix length is unchanged, so sending only one camera also works (the `missing_cameras` field in the response lists the missing slots).
- Resizing is **224x224 center padding** (`pad_mode=center`): aspect ratio preserved, padding centered; smolvla is top-left padding and fastwam stretches directly.
- Two optimizations for empty slots: `--skip-empty-cams` (the eager path skips the ViT entirely, bit-for-bit identical) and `--pad-free` (removes the placeholder slots from the prefix, usable on both eager and graph).
- `--camera-alias SRC=DST[,SRC=DST...]`: SRC is the key sent by the client and DST is the checkpoint's image key (e.g. `image2`); renaming happens only when the frame lacks DST, and multiple mappings go in the same comma-separated value.
- `state` is concatenated with `task` into the text prompt (`Task: ..., State: <discretized state>;\nAction: `), and the client just sends the state array as usual.

## Arguments that do not belong to this model

- `--no-expert-prefix-kv-cache`: **not a pi05 argument** (the engine accepts it but does not use it, purely for CLI uniformity): upstream lerobot's pi05 has no such switch (there is no `use_cache` in `configuration_pi05.py`), and prefix KV reuse is **structural** — prefill builds the prefix KV cache with `use_cache=True`, and every denoising step carries it along and reuses it with `use_cache=False` (read-only, only the suffix is computed), so there is nothing to turn off.
- smolvla-specific: `--tl-llm-fused-attn`, `--tl-vit-oproj` → `NotImplementedError`.
- fastwam-specific: `--pack-qkv`, `--no-action-context-cache`, `--no-prompt-cache`, `--cu-fused-*`, `--require-fused`, `--action-pre-fused`, `--video-pre-fused` → `NotImplementedError`.
- fastwam sidecar: `--text-encoder-dir`, `--vae-dir`, `--text-encoder-device` (when not `cpu`), `--text-emb-cpu`, `--video-*`, `--action-fp8`, `--text-fp8` → `NotImplementedError`.
