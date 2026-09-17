# tybok_gateway_cpp — C++ WebSocket gateway

[English](README.md) | [简体中文](README.zh-CN.md)

C++ implementation equivalent to the Python gateway (`tybok/gateway.py`): WebSocket frontend + fused
decode→resize→encode pipeline + true multithreaded orchestration (reader thread + decode thread pool +
cross-request lookahead), using the same IPC frame protocol as the inference worker (`tybok worker`,
Python) (`protocol.py` byte-exact compatible).
No torch dependency (only CUDA driver API + libjpeg/libpng + system headers).

## Build

```bash
cmake -S gateway_cpp -B gateway_cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build gateway_cpp/build -j8
```

Artifact: `gateway_cpp/build/tybok_gateway_cpp`.

Notes:

- `image.cpp` must be built with `-mavx2 -mfma -ffp-contract=fast` (CMakeLists already sets this
  separately for `src/image.cpp`) — the compile-time prerequisite for resize to be **bit-exact**
  with torch `F.interpolate(bilinear, align_corners=False)` + `F.pad`; dropping it drifts by 1 ulp.
- Depends only on the CUDA **driver** API (`cuMemAlloc`/`cuMemcpyHtoDAsync`/`cuIpcGetMemHandle`/`cuEvent`),
  does not link torch.

## Run

```bash
# 1) start the inference worker first (Python)
python -m tybok worker --model <ckpt> --graph --socket /tmp/tybok_worker.sock

# 2) start the C++ gateway
./gateway_cpp/build/tybok_gateway_cpp --worker-socket /tmp/tybok_worker.sock --port 8765 [--gpu-direct]
```

Arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--worker-socket PATH` | `/tmp/tybok_worker.sock` | worker Unix socket |
| `--host HOST` | `0.0.0.0` | bind address |
| `--port PORT` | `8765` | WebSocket port |
| `--gpu-direct` | off | enable GPU direct transfer (HtoD + cudaIpcMemHandle zero-copy IPC); byte transport by default |
| `--gpu-device N` | `0` | device used for GPU direct transfer (only effective with `--gpu-direct`) |
| `--threads N` | hardware thread count | decode/resize thread pool size |
| `--timing` | off | print per-request stage timings (stderr) |

## Two transports (aligned with the Python gateway; GPU direct is optional)

- **Byte transport (default)**: the same inline-payload frame as the Python gateway's `encode_request`
  (`[4B total][4B hlen][JSON header][float32 image/state payload]`), with a zero-copy `np.frombuffer`
  view on the worker side. Does not initialize CUDA, so it runs directly in environments without a GPU.
- **GPU direct transfer (`--gpu-direct`)**: the frame carries no tensor bytes; each tensor spec carries
  base64 `cudaIpcMemHandle` import metadata; the worker imports it zero-copy with
  `torch.UntypedStorage._new_shared_cuda`. On GPU initialization failure it **exits directly** (no silent fallback).

The pixels seen by the two transports are bit-exact, and are also bit-exact with the Python gateway's two transports.

## Verification

```bash
# resize is bit-exact with torch (CMake target test_image): read raw RGB, write float32 CHW
gateway_cpp/build/test_image IN.rgb H W TARGET_W TARGET_H OUT.f32
```

## Known caveats

- Before multi-process tests, first `pkill -f 'tybok worker'` / `pkill -f tybok_gateway_cpp` and delete
  `/tmp/vla_*.sock*`, otherwise ports/sockets conflict.
- The GPU direct transfer's source tensor lifetime covers the entire worker roundtrip (`finalize`
  holds it in `unique_ptr<Request>` function scope, destructed after the roundtrip) — do not move it into an `if`
  block scope, otherwise the worker opens already-freed memory (`CUDA error: invalid argument`).
