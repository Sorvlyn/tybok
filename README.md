# TyBoK (`tybok`)

[English](README.md) | [简体中文](README.zh-CN.md)

A lightweight deployment engine for VLA / WAM models. Ships with **SmolVLA**, **pi0.5** and **fastWAM**.

## Features

- **Model implementations without heavy dependencies**: built on PyTorch `nn.Module` and following the lerobot implementations, with weights loaded straight from `safetensors`; supports SmolVLA / pi0.5 / fastWAM. At runtime only `torch` / `safetensors` / `tokenizers` / `numpy` / `triton` are needed.
- **Two gateway implementations, Python and C++**: the Python gateway (`aiohttp` + `Pillow` + `torch`, `tybok/gateway.py`) and the C++ gateway (`gateway_cpp/`, multithreaded orchestration, depending only on the CUDA driver API + libjpeg/libpng, with no torch linkage). The two are numerically isomorphic and can be swapped to suit the deployment environment.
- **Multi-model architecture**: the gateway / worker / protocol layers depend only on the `PolicyEngine` interface; each model is a `tybok/policies/<name>/` package registered through `@register`, so the gateway needs zero changes (cameras, resolution, action dimension and other metadata are obtained automatically from the `describe` message).

## Installation

**One-shot install (Python components + C++ gateway, recommended)**:

```bash
git clone https://github.com/Sorvlyn/tybok.git
cd tybok

# System dependencies (Ubuntu/Debian, one-off): g++ cmake libjpeg-dev libpng-dev + CUDA toolkit
sudo apt-get install -y g++ cmake libjpeg-dev libpng-dev

bash scripts/install.sh                    # use the current python environment: install package deps + build the C++ gateway
bash scripts/install.sh --skip-cpp         # Python components only (worker-only deployment)
```

The one-shot installer [`scripts/install.sh`](scripts/install.sh) auto-detects the CUDA toolkit (`/usr/local/cuda-13.2`, `/usr/local/cuda-12.8`; override with `--cuda-root` or the `CUDA_ROOT` environment variable).

**Manual step-by-step install**:

```bash
# Run from this project's directory (i.e. tybok/)
cd tybok

# Option 1: run straight from the source directory (no install needed)
python -m tybok --help


# Option 2: pip install
pip install -e ".[gateway]"        # base dependencies (inference) + gateway extra (aiohttp / pillow)
pip install -e .                   # base dependencies only (inference; single-machine worker deployment)

# C++ gateway (optional component, built separately)
cmake -S gateway_cpp -B gateway_cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-13.2
cmake --build gateway_cpp/build -j8
```

> Recommended environment: Python ≥ 3.10, PyTorch ≥ 2.10, CUDA 12.8 or 13.2

## Quick start

```bash
cd tybok

# One command brings up worker + gateway (default ws://0.0.0.0:8765/ws)
# --graph enables CUDA Graph; --compile uses torch.compile
# --graph-cameras 2,3: pre-capture two graphs, for 2 and 3 cameras, at startup
python -m tybok serve \
    --model /path/to/smolvla \
    --socket /tmp/tybok_worker.sock --port 8765 --graph --graph-cameras 2,3

# Or deploy the two separately
python -m tybok worker --model ... --socket /tmp/tybok_worker.sock --graph
python -m tybok gateway --worker-socket /tmp/tybok_worker.sock --port 8765

# List the shipped models
python -m tybok models
```

### Client protocol (WebSocket JSON text frames)

```json
{
  "images": { "camera1": "<base64 jpeg/png>", "camera3": "<base64 jpeg/png>" },
  "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "task": "pick up the cup",
  "mode": "select_action"
}
```

- The keys of `images` are the camera names declared by the model config (the checkpoint's `config.json`): SmolVLA uses `camera1` / `camera2` / `camera3`, pi0.5 uses `image` / `image2`, and fastWAM uses the full key name `observation.images.image` — a camera whose name does not match is treated as missing and never used (use `--camera-alias` to rename a client key to a slot);
- `state` is the proprio vector of that checkpoint (the example above is SmolVLA's 6 dimensions; pi0.5 has 8 and fastWAM has 14), with the dimension defined by the model config;
- `mode` (optional): `"select_action"` (default, returns a single action each time — the server maintains a chunk queue and re-runs inference only when the queue is empty) or `"predict_action_chunk"` (returns the whole chunk at once);
- `noise: "zeros"` (optional): deterministic zero noise, for test reproducibility.

Response: `{"ok": true, "model": "smolvla" | "pi05" | "fastwam", "mode": "...", "action": [...], "shape": [...]}`.


### Example client (Python, [`examples/client.py`](examples/client.py))

```bash
cd tybok
python examples/client.py \
    --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0 \
    --image camera1=frame1.jpg,camera3=frame3.jpg
```

- `--image CAM=FILE[,CAM=FILE...]` (a single comma-separated value, not repeatable — the same convention as the server's `--camera-alias`): `CAM` is a camera name from the model config (in the example above SmolVLA's `camera1` / `camera3`; for pi0.5 use `image` / `image2`), and `--state` is the proprio dimension of that checkpoint;
- For the client's remaining arguments (`--chunk` / `--from-ref` / `--noise-zero` / `--frames`) see "Arguments" in [`docs/usage.md`](docs/usage.md).

> The client is an example, not a server-side component (the `tybok` package contains only the server side: worker / gateway / serve / validate).

## More usage

- **Model arguments** (checkpoint resolution, the fused optimization tiers `--tl-*` / `--cu-fused-*`, sampling and diagnostics, camera slots, ...) are documented in each backend's subdirectory README:
  [smolvla](tybok/policies/smolvla/README.md) · [pi05](tybok/policies/pi05/README.md) · [fastwam](tybok/policies/fastwam/README.md);
- **Arguments beyond the model arguments** (worker / gateway / client / IPC transport) and other usages (in-process quick inference, consistency validation, inference performance optimization) are in [`docs/usage.md`](docs/usage.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Some code is derived from Apache-2.0 upstream projects (LeRobot, diffusers, openpi, Wan2.2); see [`NOTICE`](NOTICE) for the individual copyright notices and the affected modules.
