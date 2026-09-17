# 使用文档

[English](usage.md) | [简体中文](usage.zh-CN.md)

本文档列出 worker / 网关参数，以及进程内快速推理、一致性验证与推理性能优化（CUDA Graph）的使用方法。

## 参数

模型专属参数（融合档 `--tl-*` / `--cu-fused-*`、fastWAM 的 fp8 档、RTC 等）见各后端子目录的 README；`serve` 是 worker + gateway 两个子进程的组合入口，两边的参数接在同一命令行上。

**入口与通道**

| 参数 | 默认 | 归属 | 说明 |
| --- | --- | --- | --- |
| `--model PATH` | — | `serve` / `worker` / `validate` | checkpoint 目录，**只接受本地路径**：不支持远程下载（给 Hugging Face repo id / URL 会报文件找不到）；它引用的子目录（tokenizer / VLM / VAE）同样要在本地 |
| `--model-type NAME` | 自动 | `serve` / `worker` / `validate` | 后端覆盖（`smolvla` / `pi05` / `fastwam`），默认读 checkpoint `config.json` 的 `type` |
| `--device DEV` | `auto` | `serve` / `worker` / `validate` | `auto` / `cuda` / `cuda:N` / `cpu` |
| `--socket PATH` | `/tmp/tybok_worker.sock` | `serve` / `worker` | worker 的 Unix socket（`python -m tybok.worker` 不传则走 TCP） |
| `--host` / `--port` | `0.0.0.0` / `8765` | `serve` / `gateway` | 网关绑定地址与 WebSocket 端口（`ws://HOST:PORT/ws`，健康检查 `/health`） |
| `--worker-socket PATH` | `/tmp/tybok_worker.sock` | `gateway` | 网关连接的 worker 通道 |
| `--worker-host` / `--worker-port` | `127.0.0.1` / `5555` | `gateway` | worker 走 TCP 时网关连接的地址与端口 |
| `--camera-alias SRC=DST[,SRC=DST...]` | 关闭 | `serve` / `worker` / `gateway` / `validate` | 把客户端相机键 `SRC` 改名到 checkpoint 的槽位 `DST`（只在该槽位缺失时生效） |

**IPC 与管线化**（效果与实测见「推理性能优化」）

| 参数 | 默认 | 归属 | 说明 |
| --- | --- | --- | --- |
| `--shm-ipc` | 关闭 | `serve` / `worker` / `gateway` | 张量载荷走共享内存环形槽，socket 只传 ~1KB JSON header（逐字节一致） |
| `--gpu-ipc` | 关闭 | `serve` / `gateway` | GPU 直传：网关 HtoD 后按 cudaIpcMemHandle 共享给 worker（逐字节一致；需 CUDA，与 `--shm-ipc` 互斥） |
| `--gpu-slots N` | `8` | `serve` / `gateway` | `--gpu-ipc` 的 CUDA keep-alive 环深度 |
| `--ipc-uint8` | 关闭 | `serve` / `gateway` | 图像量化成 uint8 传输（载荷小 4 倍，1/255 舍入——破坏逐位一致） |
| `--max-inflight N` | `4` | `serve` / `gateway` | 每连接允许超前解码的流水线消息数 |
| `--timing` | 关闭 | `gateway` | 每请求打一行 `[timing]`（解码等待 + 端到端往返）；与 worker 侧启动一次的 `--profile` 互补 |

**客户端（`examples/client.py`）**（`--url` / `--task` / `--state` / `--image` 见主目录 README 的「示例客户端」）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--chunk` | 关闭 | 请求整个动作 chunk（即 `mode: "predict_action_chunk"`） |
| `--from-ref PATH` | 关闭 | 用一份参考帧（`.pt`，含 `observation.images.*` 与 `observation.state`）构造请求，确定性回放 |
| `--noise-zero` | 关闭 | 去噪噪声取零（`noise: "zeros"`），输出可与参考逐位比对 |
| `--frames N` | `1` | 连续发 N 帧（用于验证 chunk 队列语义） |
| `--rtc*` | 关闭 | RTC 请求字段（`--rtc` / `--rtc-prev` / `--rtc-delay` / `--rtc-horizon`），需 worker 以 `--rtc` 启动，见 `pi05` 子目录 README |

**C++ 网关**（`gateway_cpp/build/tybok_gateway_cpp`）：`--worker-socket` / `--host` / `--port` / `--timing` 与 Python 网关同名同义；另有 `--gpu-direct`（GPU 直传，对应 Python 网关的 `--gpu-ipc`）、`--gpu-device N`、`--threads N`。传输细节见「推理性能优化」的 C++ 网关小节。

## 进程内快速推理（调试，无需启动服务）

直接加载引擎、跑一次推理、打印结果，方便打断点、看日志、profile 或快速验证 checkpoint：

```bash
cd tybok

# 回放参考帧（--noise-zero 确定性，chunk[0] 与参考输出一致）
python examples/run_inference.py --checkpoint <ckpt> --frame <frame.pt> --noise-zero --chunk

# 合成帧（零图像 + 指定 state/task）跑单个动作
python examples/run_inference.py --checkpoint <ckpt> --state 0,0,0,0,0,0 --task "close the door"

# 验证动作队列语义（chunk 内连续取步，队空才重新推理）
python examples/run_inference.py --checkpoint <ckpt> --frame <frame.pt> --noise-zero --steps 5
```

## 一致性验证

不开模型侧融合档时，各组件（含完整动作 chunk）与 lerobot 参考输出逐位一致；`--graph` 与专家前缀 KV 缓存（默认开）同样保持逐位一致。融合档与 `--compile` 属 drift 档，会改变数值。

## 推理性能优化（CUDA Graph）

以下按优化项给出实测与取舍（延迟为单请求 `predict_action_chunk`）：

- **`--graph`**：把视觉编码 + prefix prefill + 去噪拆成**两张 CUDA Graph**（prefill 图 + 去噪图，配合预分配 KV buffer、
  常量预计算与 bucketize GPU 化），消除约 5000 次 kernel 启动开销，**保持逐位一致**；
  真实 checkpoint 实测：3 相机 eager 100.6ms → graph 45.7ms（2.2×）；**2 相机主场景 95.7ms → 35.8ms（2.5×）**；
  **2 相机与 3 相机各一组图**，`--graph-cameras 2,3` 可在启动时预捕获（零请求捕获延迟），其它相机数懒捕获；
- **专家前缀 KV 缓存（默认开启，`--no-expert-prefix-kv-cache` 关闭）**：交叉注意力层的 expert `k_proj`/`v_proj`（fp32）
  作用在 prefill 后**不变**的 VLM 前缀 KV 上，原实现每步重算 8 层 × 10 步 = 80 次；现在 prefill 后算一次，
  去噪图直接读缓存 buffer（值逐位一致）。实测 graph 省 ~0.3ms、eager 省 ~1ms（投影 GEMM 很小）；
  `--compile` 下引擎在编译**前**临时关掉它（它是逐请求变化的状态，dynamo 会对它装 guard、导致区域反复重新特化），
  编译/预热/图捕获完成后按设置恢复；两处的开关状态都会打 INFO；
- **`--compile`**：torch.compile（含 TF32），约 29.3ms，**改数值**（不再与参考输出对齐）且**冷编译约 3 分钟**
  （第二次起命中缓存 ~13s）；`--compile --fast` 更慢的冷启动（~4.5 分钟）且漂移更大（chunk_post.max 0.057）——
  一般不建议部署用 compile 档；
- **`--steps N`**：覆盖去噪步数（默认 10）。Euler 每步约 2.4ms：8 步 33.3ms（误差 0.14）、
  6 步 28.5ms（误差 0.35）、5 步 25.9ms（误差 0.39）——**减步对输出的影响大于模型侧融合档**
  （后者 10 步误差仅 0.03），低于 8 步建议配合采样器升级/蒸馏；
- **`--sampler heun`**：二阶 predictor-corrector（Heun），N 步 = 2N 次速度求值。
  **以 Euler-10 参考为基准**（2 相机 graph）：Heun-5 36.6ms 误差 0.11、Heun-4 32.1ms 误差 0.23、
  Heun-3 误差 0.50。同步数下 Heun 比 Euler 准得多（Heun-5 0.11 vs Euler-5 0.39），
  但步数越多 Heun 越逼近真实 ODE 解（离 Euler-10 参考约 0.35，即 Euler-10 自身的离散化误差）——
  若目标是贴近 Euler-10 参考，Euler-8（0.14）仍是性价比最优；Heun 的价值在**同等算力下更准的真解**；
- **`--seed N`**：为去噪噪声提供独立 RNG（与全局 torch RNG 隔离），同一 seed 产生可复现的动作序列；
  与 `--graph` / `--compile` / `--sampler` 及模型侧融合档均兼容（graph 模式噪声在图外生成再拷入静态 buffer）；
  不传则保持历史行为（全局 RNG）。
- **`--profile`**：worker/serve 启动时跑一次延迟剖析并打印到日志（`[tybok] profile` 前缀）：
  按当前配置（graph/sampler/steps/相机数）测 `predict_action_chunk` 的 wall 时间；smolvla 额外给出
  阶段分解（阶段名 = 被计时的那条模型原语：`prepare_images` / `embed_prefix` /
  `vlm_with_expert.forward` / `_denoise_loop` / `postprocessor`），pi05 / fastwam 只有
  wall 时间（三个后端都支持该参数）。graph 模式下 wall 为图回放路径，
  阶段表为 eager 组件耗时（图消除的是 launch 开销），可直接看到图的收益；报告带 `flush=True`，
  即使 worker 随后被终止也能完整落盘。也可在代码里调用 `engine.profile(cameras=2)` 按需剖析。
- **网关侧优化（已实现）**：
  - **图像解码 offload 线程池 + 按相机并行**（`gateway.py::_process_images`）：JPEG 解码是网关最大
    单请求成本（~5-8ms/相机，CPU）；原来在 aiohttp 事件循环内同步执行（多客户端互相阻塞）。现在
    每相机 `asyncio.to_thread` 并行（PIL 解码与 torch resize 释放 GIL，真并行），实测 2 相机
    12.8ms → **3.6ms（3.6×）**，且事件循环不再被阻塞；
  - **worker 持久连接**（`WorkerClient`）：请求间复用同一 socket + 锁串行（worker 推理本就串行），
    免去每请求 connect/accept/线程创建（~0.1-0.5ms），worker 线程数恒定；失败自动重连；
  - **IPC 零拷贝解析**（`protocol.py::decode_request`）：去掉 6.3MB 帧的防御性 `.copy()`（视图在
    `handle_frame` 全程存活），省 ~0.3ms/请求；
  - **torchvision 解码加速**（`image_utils.py`）：JPEG/PNG 优先走 torchvision 解码器
    （libjpeg-turbo，直接 CHW 输出免转置，~1.3×），Pillow 自动回退；
  - **跨请求管线化**：网关每消息一个 task（`--max-inflight`，默认 4）——流水线客户端（机器人
    下一帧不等上一动作）的下一帧解码与当前推理重叠；worker 拆读线程（recv 与推理重叠）。实测
    流水线客户端吞吐 +17%（48.5 → 41.6ms/req，serve --graph 2 相机）；
  - **IPC 序列化单缓冲重写**（`protocol.py`）：`encode_request` 原来每 tensor 一次 `tobytes()`
    中间块 + `join` + 帧拼接，实测 ~5ms/请求（多段大分配组合会触发 glibc 慢速态，纯 CPU 微基准
    会掩盖）；改为全部 tensor 经可写 numpy 视图拷入一次预分配的 bytearray 并**原样返回**（末尾
    再 `bytes()` 拷贝会回到 ~3.6ms），实测 **5.0 → 0.39ms（13×）**；worker `_read_frame` 改
    `recv_into` 单缓冲，2.7 → 1.27ms；`decode_request` 带绝对偏移零拷贝读整帧（免 payload 切片
    拷贝）；逐字节一致，数值零影响；
  - **共享内存 IPC（`--shm-ipc`，opt-in）**：~6MB 的 tensor payload 经 mmap 环形槽（8 槽）
    直传，socket 只走 ~1KB JSON header（带 `"shm":{"slot":k}`），worker `np.frombuffer` 零拷贝
    读槽；槽大小由 worker 随 `describe` 下发，payload 超槽自动回退帧内路径；实测流水线客户端
    **41.6 → 38.0ms/req（+9%）**、误差仍 0.0。注意：`mmap.mmap(fd, size, access=...)` 必须用
    关键字传 `access`——位置参数会落进 `flags`，`ACCESS_WRITE`(=2)=`MAP_PRIVATE`，写不跨进程可见；
  - **GPU 直传（`--gpu-ipc`，opt-in）**：网关把帧 HtoD 到自己的 CUDA 缓冲，socket 只传 JSON
    header（含每个 tensor 的 cudaIpcMemHandle 元数据），worker 用
    `UntypedStorage._new_shared_cuda` 零拷贝导入（torch.multiprocessing 同款私有 API）后 D2D
    拷进图静态缓冲；网关侧 8 槽 keep-alive 环保证 source 存活；`_share_cuda_` 的事件同步保证
    不读半写缓冲；与 `--shm-ipc` 互斥、与 `--ipc-uint8` 可叠加；实测流水线 **38.0ms/req**、
    顺序 **44.9ms/req**（worker 侧 recv+HtoD 被隐藏）、误差仍 **0.0**。代价：网关需持有 CUDA
    上下文（~0.5-2GB 显存，启动时初始化）；私有 API 无公开版本保证；
  - **uint8 IPC 可选传输**（`--ipc-uint8`）：图像量化 uint8 传输（6.3MB → 1.6MB）；实测零噪声下
    原生 512×512 输入**零漂移**、需 resize 时漂移 0.011；bit-exact 部署保持默认 float32；
  - 端到端请求路径（serve --graph 2 相机）：解码 3.6（线程池并行，管线化隐藏）+ IPC ~1.7 +
    worker 推理外 ~0.9 + 推理 34.6 ≈ **41.6ms/req**（`--shm-ipc` / `--gpu-ipc` 下 **38.0ms/req**）。

**C++ 网关（`gateway_cpp/`，与 Python 网关等价的 WebSocket 网关）**：

- 构建/运行（无 torch 依赖，仅 CUDA driver API + libjpeg/libpng）：
  ```bash
  cmake -S gateway_cpp -B gateway_cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
  cmake --build gateway_cpp/build -j8
  ./gateway_cpp/build/tybok_gateway_cpp --worker-socket /tmp/tybok_worker.sock --port 8765 [--gpu-direct]
  ```
- **传输可选项（与 Python 网关对齐，`--gpu-direct` 默认关闭）**：
  - **字节传输（默认）**：与 Python 网关 `encode_request` 同款的 inline-payload 帧
    （`[4B total][4B hlen][JSON header][float32 图像/state 载荷]`），worker 端 `np.frombuffer`
    零拷贝视图，逐字节一致；无 GPU 环境可用（不初始化 CUDA，启动不需要 `--gpu-direct` 也能跑）；
  - **GPU 直传（`--gpu-direct`，opt-in）**：HtoD + `cudaIpcMemHandle` 零拷贝 IPC（Python 网关
    `--gpu-ipc` 的 C++ 等价物），socket 只走 JSON header；需 CUDA 环境，启动时 `cuCtxCreate`
    （占用少量显存）；GPU 初始化失败时直接退出（不会静默降级）；
  - 两种传输模型看到的像素逐位一致，与 Python 网关的两种传输也一致；
- **与 Python 网关的差异**：C++ 为多线程编排（每连接 reader 线程 + 解码线程池 + 2 层
  lookahead 管线化），Python 为 asyncio + `to_thread`；C++ 字节路径与 Python 字节路径数值同构，
  GPU 路径与 Python `--gpu-ipc` 数值同构；实测（2 相机 graph）：C++ GPU
  直传流水线 **38ms/req**、多客户端（16 并发）网关 CPU **<4%**（Python 网关 ~18%）；
- `--graph` 与 `--compile` 可叠加（`--compile` 会由引擎自动关掉 `--overlap` 并打一条 INFO；逐位一致仅适用于不开 `--compile` 的 graph）；
  `--sampler` / `--steps` / `--profile` 及模型侧融合档与两者正交可叠加；
  CPU / 捕获失败时 `--graph` 自动降级为 eager。

## 安全边界（现状说明）

**本版本没有鉴权，也没有 TLS。** 网关的 `/ws`（推理）和 `/health`（探活）都是明文 WebSocket，任何能连上
该端口的一方都可以提交观测、取回动作；代码里没有 `ssl_context`、token 或任何访问控制。C++ 网关
（`gateway_cpp/`）同样没有鉴权/TLS，两者在这点上等价。

因此不要把网关端口暴露到不可信网络：

- `--host` 默认 `0.0.0.0`（监听所有网卡），受信内网之外建议显式收到具体网卡或 `127.0.0.1`；
- 需要跨机访问时，用 SSH 隧道（`ssh -L 8765:127.0.0.1:8765 <user>@<host>`）或带 TLS + 认证的反向代理
  （nginx / Caddy）挡在前面；
- 用防火墙限制来源 IP。

网关与 worker 之间的 IPC 也没有鉴权：默认的 Unix socket 靠文件系统权限（按当前 umask 创建，实测
`srwxrwxr-x`），因此两者应以同一用户运行；`/health` 无需凭证即可探测模型类型与相机列表。

这是**已知现状**，不是可开关的配置项——要对公网提供服务，必须在部署层（隧道 / 反向代理 / 防火墙）补齐。
