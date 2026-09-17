# Usage

[English](usage.md) | [简体中文](usage.zh-CN.md)

This document lists the worker / gateway arguments, and how to use in-process quick inference, consistency validation and inference performance optimization (CUDA Graph).

## Arguments

Model-specific arguments (the fused tiers `--tl-*` / `--cu-fused-*`, fastWAM's fp8 tier, RTC, ...) are documented in each backend subdirectory's README; `serve` is the combined entry point for the worker + gateway processes, and the arguments of both ends are accepted on the same command line.

**Entry points and channels**

| Argument | Default | Owner | Description |
| --- | --- | --- | --- |
| `--model PATH` | — | `serve` / `worker` / `validate` | checkpoint directory, **local path only**: remote download is not supported (a Hugging Face repo id / URL fails as file-not-found); the subdirectories it references (tokenizer / VLM / VAE) must be local as well |
| `--model-type NAME` | automatic | `serve` / `worker` / `validate` | backend override (`smolvla` / `pi05` / `fastwam`); by default reads `type` from the checkpoint `config.json` |
| `--device DEV` | `auto` | `serve` / `worker` / `validate` | `auto` / `cuda` / `cuda:N` / `cpu` |
| `--socket PATH` | `/tmp/tybok_worker.sock` | `serve` / `worker` | the worker's Unix socket (`python -m tybok.worker` falls back to TCP if unset) |
| `--host` / `--port` | `0.0.0.0` / `8765` | `serve` / `gateway` | gateway bind address and WebSocket port (`ws://HOST:PORT/ws`, health check `/health`) |
| `--worker-socket PATH` | `/tmp/tybok_worker.sock` | `gateway` | the worker channel the gateway connects to |
| `--worker-host` / `--worker-port` | `127.0.0.1` / `5555` | `gateway` | address and port the gateway connects to when the worker uses TCP |
| `--camera-alias SRC=DST[,SRC=DST...]` | off | `serve` / `worker` / `gateway` / `validate` | rename the client camera key `SRC` to the checkpoint slot `DST` (only takes effect when that slot is missing) |

**IPC and pipelining** (results and measurements in "Inference performance optimization")

| Argument | Default | Owner | Description |
| --- | --- | --- | --- |
| `--shm-ipc` | off | `serve` / `worker` / `gateway` | tensor payloads travel through shared-memory ring slots, and the socket only carries a ~1KB JSON header (byte-exact) |
| `--gpu-ipc` | off | `serve` / `gateway` | GPU direct transfer: the gateway does HtoD, then shares with the worker via cudaIpcMemHandle (byte-exact; requires CUDA, mutually exclusive with `--shm-ipc`) |
| `--gpu-slots N` | `8` | `serve` / `gateway` | depth of the CUDA keep-alive ring for `--gpu-ipc` |
| `--ipc-uint8` | off | `serve` / `gateway` | quantize images to uint8 for transfer (payload 4× smaller, 1/255 rounding — breaks bit-exactness) |
| `--max-inflight N` | `4` | `serve` / `gateway` | number of pipelined messages per connection allowed for lookahead decoding |
| `--timing` | off | `gateway` | print one `[timing]` line per request (decode wait + end-to-end round trip); complements the worker-side `--profile`, which runs once at startup |

**Client (`examples/client.py`)** (`--url` / `--task` / `--state` / `--image` are documented in "Example client" in the root README)

| Argument | Default | Description |
| --- | --- | --- |
| `--chunk` | off | request the whole action chunk (i.e. `mode: "predict_action_chunk"`) |
| `--from-ref PATH` | off | build the request from a reference frame (`.pt`, containing `observation.images.*` and `observation.state`) for deterministic replay |
| `--noise-zero` | off | use zero denoising noise (`noise: "zeros"`), so the output can be compared bit-for-bit with the reference |
| `--frames N` | `1` | send N frames in a row (for validating chunk queue semantics) |
| `--rtc*` | off | RTC request fields (`--rtc` / `--rtc-prev` / `--rtc-delay` / `--rtc-horizon`); the worker must be started with `--rtc`, see the `pi05` subdirectory README |

**C++ gateway** (`gateway_cpp/build/tybok_gateway_cpp`): `--worker-socket` / `--host` / `--port` / `--timing` have the same names and meanings as the Python gateway; it also has `--gpu-direct` (GPU direct transfer, corresponding to the Python gateway's `--gpu-ipc`), `--gpu-device N` and `--threads N`. Transport details are in the C++ gateway subsection of "Inference performance optimization".

## In-process quick inference (debugging, no server needed)

Load the engine directly, run one inference and print the result — convenient for breakpoints, reading logs, profiling or a quick sanity check of a checkpoint:

```bash
cd TyBoK

# Replay a reference frame (--noise-zero is deterministic, chunk[0] matches the reference output)
python examples/run_inference.py --checkpoint <ckpt> --frame <frame.pt> --noise-zero --chunk

# Synthetic frame (zero images + given state/task) to run a single action
python examples/run_inference.py --checkpoint <ckpt> --state 0,0,0,0,0,0 --task "close the door"

# Validate action queue semantics (take consecutive steps inside the chunk, re-run inference only when the queue is empty)
python examples/run_inference.py --checkpoint <ckpt> --frame <frame.pt> --noise-zero --steps 5
```

## Consistency validation

With no model-side fused tier enabled, every component (including the full action chunk) is bit-exact with the lerobot reference output; `--graph` and the expert prefix KV cache (on by default) likewise stay bit-exact. The fused tiers and `--compile` are drift tiers and do change the values.

## Inference performance optimization (CUDA Graph)

Below are the measured results and trade-offs for each optimization (latency is for a single `predict_action_chunk` request):

- **`--graph`**: splits vision encoding + prefix prefill + denoising into **two CUDA Graphs** (a prefill graph + a denoising graph, together with a preallocated KV buffer,
  constant precomputation and GPU-side bucketize), eliminating ~5000 kernel launches while **staying bit-exact**;
  measured on a real checkpoint: 3 cameras eager 100.6ms → graph 45.7ms (2.2×); **2 cameras, the main scenario, 95.7ms → 35.8ms (2.5×)**;
  **one graph set each for 2 cameras and 3 cameras**, and `--graph-cameras 2,3` pre-captures them at startup (zero per-request capture latency); other camera counts are captured lazily;
- **Expert prefix KV cache (on by default; disable with `--no-expert-prefix-kv-cache`)**: the expert `k_proj`/`v_proj` (fp32) of the cross-attention layers
  act on the VLM prefix KV that is **unchanged** after prefill; the original implementation recomputed it every step, 8 layers × 10 steps = 80 times, whereas now it is computed once after
  prefill and the denoising graph reads the cache buffer directly (values bit-exact). Measured: graph saves ~0.3ms and eager saves ~1ms (the projection GEMMs are tiny);
  under `--compile` the engine temporarily turns it off **before** compiling (it is per-request-varying state, and dynamo would install a guard on it, causing the region to be re-specialized repeatedly),
  then restores it per the setting once compile / warmup / graph capture is done; both switch states are logged at INFO;
- **`--compile`**: torch.compile (including TF32), about 29.3ms, **changes the values** (no longer aligned with the reference output) and **cold compilation takes about 3 minutes**
  (a cache hit from the second run on, ~13s); `--compile --fast` has an even slower cold start (~4.5 minutes) and larger drift (chunk_post.max 0.057) —
  the compile tier is generally not recommended for deployment;
- **`--steps N`**: overrides the number of denoising steps (default 10). Euler costs about 2.4ms per step: 8 steps 33.3ms (error 0.14),
  6 steps 28.5ms (error 0.35), 5 steps 25.9ms (error 0.39) — **reducing the step count affects the output more than a model-side fused tier**
  (which has an error of only 0.03 at 10 steps); below 8 steps it is advisable to pair it with a sampler upgrade / distillation;
- **`--sampler heun`**: second-order predictor-corrector (Heun), N steps = 2N velocity evaluations.
  **With Euler-10 as the reference** (2 cameras, graph): Heun-5 36.6ms error 0.11, Heun-4 32.1ms error 0.23,
  Heun-3 error 0.50. At equal step counts Heun is far more accurate than Euler (Heun-5 0.11 vs Euler-5 0.39),
  but with more steps Heun gets closer to the true ODE solution (about 0.35 away from the Euler-10 reference, i.e. Euler-10's own discretization error) —
  if the goal is to stay close to the Euler-10 reference, Euler-8 (0.14) is still the best value; Heun's value is the **more accurate true solution at the same compute**;
- **`--seed N`**: provides an independent RNG for the denoising noise (isolated from the global torch RNG); the same seed produces a reproducible action sequence;
  compatible with `--graph` / `--compile` / `--sampler` and the model-side fused tiers (in graph mode the noise is generated outside the graph and then copied into a static buffer);
  if unset, the historical behavior is kept (global RNG).
- **`--profile`**: at worker/serve startup, runs one latency profiling pass and prints it to the log (`[tybok] profile` prefix):
  it measures the wall time of `predict_action_chunk` under the current configuration (graph/sampler/steps/camera count); for smolvla it additionally gives a
  stage breakdown (the stage names are the timed model primitives: `prepare_images` / `embed_prefix` /
  `vlm_with_expert.forward` / `_denoise_loop` / `postprocessor`), while pi05 / fastwam only have the
  wall time (all three backends support this argument). In graph mode the wall time is for the graph replay path
  and the stage table lists the eager component timings (what the graph eliminates is launch overhead), so the graph's benefit is directly visible; the report is written with `flush=True`,
  so it lands on disk completely even if the worker is killed right afterwards. You can also call `engine.profile(cameras=2)` in code to profile on demand.
- **Gateway-side optimizations (implemented)**:
  - **Image decoding offload thread pool + per-camera parallelism** (`gateway.py::_process_images`): JPEG decoding is the gateway's largest
    single-request cost (~5-8ms/camera, CPU); it used to run synchronously inside the aiohttp event loop (multiple clients blocking each other). Now
    each camera runs in parallel via `asyncio.to_thread` (PIL decoding and torch resize release the GIL, so it is genuinely parallel): measured 2 cameras
    12.8ms → **3.6ms (3.6×)**, and the event loop is no longer blocked;
  - **Worker persistent connection** (`WorkerClient`): the same socket is reused across requests plus lock serialization (worker inference is serial anyway),
    avoiding per-request connect/accept/thread creation (~0.1-0.5ms) and keeping the worker thread count constant; reconnects automatically on failure;
  - **IPC zero-copy parsing** (`protocol.py::decode_request`): drops the defensive `.copy()` of 6.3MB frames (the view stays alive throughout
    `handle_frame`), saving ~0.3ms/request;
  - **torchvision decoding acceleration** (`image_utils.py`): JPEG/PNG preferentially go through the torchvision decoder
    (libjpeg-turbo, direct CHW output with no transpose, ~1.3×), with automatic fallback to Pillow;
  - **Cross-request pipelining**: one task per gateway message (`--max-inflight`, default 4) — for a pipelined client (the robot's
    next frame does not wait for the previous action) the next frame's decoding overlaps the current inference; the worker splits off a read thread (recv overlapping inference). Measured:
    pipelined client throughput +17% (48.5 → 41.6ms/req, serve --graph 2 cameras);
  - **IPC serialization rewritten around a single buffer** (`protocol.py`): `encode_request` used to do one `tobytes()`
    intermediate block per tensor + `join` + frame concatenation, measured ~5ms/request (combining many large allocations triggers glibc's slow path, which a pure-CPU microbenchmark
    hides); it now copies all tensors into one preallocated bytearray through a writable numpy view and **returns it as is** (a trailing
    `bytes()` copy would bring it back to ~3.6ms), measured **5.0 → 0.39ms (13×)**; the worker's `_read_frame` now uses
    `recv_into` with a single buffer, 2.7 → 1.27ms; `decode_request` reads the whole frame zero-copy with an absolute offset (avoiding payload slice
    copies); byte-exact, zero numerical impact;
  - **Shared-memory IPC (`--shm-ipc`, opt-in)**: the ~6MB tensor payload goes directly through mmap ring slots (8 slots),
    the socket only carries the ~1KB JSON header (with `"shm":{"slot":k}`), and the worker does a zero-copy `np.frombuffer`
    read of the slot; the slot size is sent down by the worker along with `describe`, and a payload larger than the slot falls back to the in-frame path automatically; measured pipelined client
    **41.6 → 38.0ms/req (+9%)**, error still 0.0. Note: `mmap.mmap(fd, size, access=...)` must pass `access`
    as a keyword — a positional argument lands in `flags`, and `ACCESS_WRITE`(=2)=`MAP_PRIVATE`, so writes are not visible across processes;
  - **GPU direct transfer (`--gpu-ipc`, opt-in)**: the gateway does HtoD of the frames into its own CUDA buffers, the socket carries only the JSON
    header (with cudaIpcMemHandle metadata for each tensor), and the worker imports them zero-copy with
    `UntypedStorage._new_shared_cuda` (the same private API used by torch.multiprocessing) and then D2D-copies them
    into the graph's static buffers; an 8-slot keep-alive ring on the gateway side keeps the source alive; `_share_cuda_`'s event synchronization guarantees
    that no half-written buffer is read; mutually exclusive with `--shm-ipc` and stackable with `--ipc-uint8`; measured pipelined **38.0ms/req**,
    sequential **44.9ms/req** (recv+HtoD hidden on the worker side), error still **0.0**. Cost: the gateway must hold a CUDA
    context (~0.5-2GB VRAM, initialized at startup); the private API comes with no public version guarantee;
  - **uint8 IPC optional transport** (`--ipc-uint8`): images quantized to uint8 for transfer (6.3MB → 1.6MB); measured with zero noise,
    native 512×512 input has **zero drift**, and the drift is 0.011 when a resize is needed; bit-exact deployments keep the default float32;
  - End-to-end request path (serve --graph 2 cameras): decode 3.6 (thread pool in parallel, hidden by pipelining) + IPC ~1.7 +
    outside worker inference ~0.9 + inference 34.6 ≈ **41.6ms/req** (**38.0ms/req** under `--shm-ipc` / `--gpu-ipc`).

**C++ gateway (`gateway_cpp/`, a WebSocket gateway equivalent to the Python gateway)**:

- Build/run (no torch dependency, only the CUDA driver API + libjpeg/libpng):
  ```bash
  cmake -S gateway_cpp -B gateway_cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
  cmake --build gateway_cpp/build -j8
  ./gateway_cpp/build/tybok_gateway_cpp --worker-socket /tmp/tybok_worker.sock --port 8765 [--gpu-direct]
  ```
- **Transport options (aligned with the Python gateway; `--gpu-direct` is off by default)**:
  - **Byte transport (default)**: the same inline-payload frames as the Python gateway's `encode_request`
    (`[4B total][4B hlen][JSON header][float32 image/state payload]`), with the worker taking a zero-copy `np.frombuffer`
    view, byte-exact; usable in environments without a GPU (it does not initialize CUDA, and runs without `--gpu-direct`);
  - **GPU direct transfer (`--gpu-direct`, opt-in)**: HtoD + `cudaIpcMemHandle` zero-copy IPC (the C++ equivalent of the Python gateway's
    `--gpu-ipc`), the socket carrying only the JSON header; requires a CUDA environment and `cuCtxCreate`
    at startup (uses a small amount of VRAM); it exits immediately if GPU initialization fails (no silent fallback);
  - The pixels seen by the two transport models are bit-exact, and are also identical to the Python gateway's two transports;
- **Differences from the Python gateway**: C++ uses multithreaded orchestration (one reader thread per connection + a decode thread pool + 2-layer
  lookahead pipelining), Python uses asyncio + `to_thread`; the C++ byte path is numerically isomorphic to the Python byte path, and the
  GPU path is numerically isomorphic to Python's `--gpu-ipc`; measured (2 cameras, graph): C++ GPU
  direct-transfer pipelining **38ms/req**, and gateway CPU **<4%** with many clients (16 concurrent) (Python gateway ~18%);
- `--graph` and `--compile` can be stacked (`--compile` makes the engine turn `--overlap` off automatically and log an INFO line; bit-exactness applies only to graph without `--compile`);
  `--sampler` / `--steps` / `--profile` and the model-side fused tiers are orthogonal to the two and can be stacked;
  on CPU / capture failure `--graph` automatically degrades to eager.

## Security boundary (current state)

**This version has no authentication and no TLS.** The gateway's `/ws` (inference) and `/health` (liveness) are both plaintext WebSocket, so any party that can connect to
that port can submit observations and retrieve actions; there is no `ssl_context`, token or any access control in the code. The C++ gateway
(`gateway_cpp/`) likewise has no authentication/TLS, and the two are equivalent in this respect.

So do not expose the gateway port to an untrusted network:

- `--host` defaults to `0.0.0.0` (listening on all interfaces); outside a trusted LAN, explicitly bind it to a specific interface or `127.0.0.1`;
- when cross-machine access is needed, put an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 <user>@<host>`) or a reverse proxy with TLS + authentication
  (nginx / Caddy) in front;
- use a firewall to restrict source IPs.

The IPC between the gateway and the worker has no authentication either: the default Unix socket relies on filesystem permissions (created per the
current umask, measured `srwxrwxr-x`), so the two should run as the same user; `/health` lets anyone probe the model type and camera list without credentials.

This is a **known current state**, not a switchable configuration option — to serve the public internet, it must be covered at the deployment layer (tunnel / reverse proxy / firewall).
