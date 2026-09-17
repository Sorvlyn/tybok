# TyBoK（`tybok`）

[English](README.md) | [简体中文](README.zh-CN.md)

用于 VLA / WAM 的简易部署引擎。当前内置 **SmolVLA**、**pi0.5** 与 **fastWAM** 模型。

## 特性

- **模型实现无繁重依赖**：基于 PyTorch `nn.Module`、参考 lerobot 实现，权重直接由 `safetensors` 加载，支持 SmolVLA / pi0.5 / fastWAM；运行时只需 `torch` / `safetensors` / `tokenizers` / `numpy` / `triton`。
- **网关提供 Python 与 C++ 两套实现**：Python 网关（`aiohttp` + `Pillow` + `torch`，`tybok/gateway.py`）；C++ 网关（`gateway_cpp/`，多线程编排，仅依赖 CUDA driver API + libjpeg/libpng，无 torch 链接）。两者数值同构，可按部署环境互换。
- **多模型架构**：网关/worker/协议层只依赖 `PolicyEngine` 接口；每个模型一个 `tybok/policies/<name>/` 包并通过 `@register` 注册，网关零改动（通过 `describe` 消息自动获取相机、分辨率、动作维度等元数据）。

## 安装

**一键安装（Python 组件 + C++ 网关，推荐）**：

```bash
# 系统依赖（Ubuntu/Debian，一次性）：g++ cmake libjpeg-dev libpng-dev + CUDA toolkit
sudo apt-get install -y g++ cmake libjpeg-dev libpng-dev

cd TyBoK
bash scripts/install.sh                    # 用当前 python 环境，安装包依赖 + 构建 C++ 网关
bash scripts/install.sh --skip-cpp         # 只装 Python 组件（纯 worker 部署）
```

一键安装脚本 [`scripts/install.sh`](scripts/install.sh) 自动探测 CUDA toolkit（`/usr/local/cuda-13.2`、`/usr/local/cuda-12.8`，可用 `--cuda-root` 或环境变量 `CUDA_ROOT` 覆盖）。

**手动分步安装**：

```bash
# 在本项目目录（即 TyBoK/）下操作
cd TyBoK

# 方式一：源码目录直接运行（无需安装）
python -m tybok --help


# 方式二：pip 安装
pip install -e ".[gateway]"        # 基础依赖（推理）+ gateway extra（aiohttp / pillow）
pip install -e .                   # 只装基础依赖（推理；worker 单机部署）

# C++ 网关（可选组件，单独构建）
cmake -S gateway_cpp -B gateway_cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-13.2
cmake --build gateway_cpp/build -j8
```

> 环境建议：Python ≥ 3.10，PyTorch ≥ 2.10， CUDA 12.8或13.2

## 快速开始

```bash
cd TyBoK

# 一条命令拉起 worker + gateway（默认 ws://0.0.0.0:8765/ws）
# --graph 启用 CUDA Graph；--compile 用 torch.compile
# --graph-cameras 2,3：启动时预捕获 2/3 相机两张图
python -m tybok serve \
    --model /path/to/smolvla \
    --socket /tmp/tybok_worker.sock --port 8765 --graph --graph-cameras 2,3

# 或分别部署
python -m tybok worker --model ... --socket /tmp/tybok_worker.sock --graph
python -m tybok gateway --worker-socket /tmp/tybok_worker.sock --port 8765

# 列出本安装自带的模型
python -m tybok models
```

### 客户端协议（WebSocket JSON 文本帧）

```json
{
  "images": { "camera1": "<base64 jpeg/png>", "camera3": "<base64 jpeg/png>" },
  "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "task": "pick up the cup",
  "mode": "select_action"
}
```

- `images` 的键是模型 config（checkpoint 的 `config.json`）声明的相机名：SmolVLA 为 `camera1` / `camera2` / `camera3`，pi0.5 为 `image` / `image2`，fastWAM 为完整键名 `observation.images.image`——名字对不上的相机按缺失处理，不会被用到（可用 `--camera-alias` 把客户端键改名到槽位）；
- `state` 是该 checkpoint 的 proprio 向量（上例是 SmolVLA 的 6 维；pi0.5 8 维、fastWAM 14 维），维度以模型 config 为准；
- `mode`（可选）：`"select_action"`（默认，每次返回单个动作——服务端维护 chunk 队列，队空才重新推理）或 `"predict_action_chunk"`（一次返回整个 chunk）；
- `noise: "zeros"`（可选）：确定性零噪声，用于测试复现。

响应：`{"ok": true, "model": "smolvla" | "pi05" | "fastwam", "mode": "...", "action": [...], "shape": [...]}`。


### 示例客户端（Python，[`examples/client.py`](examples/client.py)）

```bash
cd TyBoK
python examples/client.py \
    --url ws://127.0.0.1:8765/ws \
    --task "put the white mug on the left plate" \
    --state 0,0,0,0,0,0 \
    --image camera1=frame1.jpg,camera3=frame3.jpg
```

- `--image CAM=FILE[,CAM=FILE...]`（单个逗号分隔的值，不可重复——与服务端 `--camera-alias` 同一约定）：`CAM` 是模型 config 里的相机名（上例为 SmolVLA 的 `camera1` / `camera3`，pi0.5 用 `image` / `image2`），`--state` 是该 checkpoint 的 proprio 维度；
- 客户端其余参数（`--chunk` / `--from-ref` / `--noise-zero` / `--frames`）见 [`docs/usage.zh-CN.md`](docs/usage.zh-CN.md) 的「参数」。

> 客户端是示例而非服务端组件（`tybok` 包只含服务端：worker / gateway / serve / validate）。

## 更多用法

- **模型参数**（checkpoint 解析、融合档 `--tl-*` / `--cu-fused-*`、采样与诊断、相机槽位等）见各后端子目录的 README：
  [smolvla](tybok/policies/smolvla/README.zh-CN.md) · [pi05](tybok/policies/pi05/README.zh-CN.md) · [fastwam](tybok/policies/fastwam/README.zh-CN.md)；
- **模型参数之外**的参数（worker / 网关 / 客户端 / IPC 传输）以及其他用法（进程内快速推理、一致性验证、推理性能优化）见 [`docs/usage.zh-CN.md`](docs/usage.zh-CN.md)。

## License

本项目采用 [Apache License 2.0](LICENSE)。

部分代码派生自 Apache-2.0 的上游项目（LeRobot、diffusers、openpi、Wan2.2），各自的版权声明与受影响模块见 [`NOTICE`](NOTICE)。
