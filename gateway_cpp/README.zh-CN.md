# tybok_gateway_cpp — C++ WebSocket 网关

[English](README.md) | [简体中文](README.zh-CN.md)

与 Python 网关（`tybok/gateway.py`）等价的 C++ 实现：WebSocket 前端 + 融合
解码→resize→编码流水线 + 真多线程编排（reader 线程 + 解码线程池 + 跨请求 lookahead），
与推理 worker（`tybok worker`，Python）用同一套 IPC 帧协议（`protocol.py` 逐字节兼容）。
无 torch 依赖（仅 CUDA driver API + libjpeg/libpng + 系统头）。

## 构建

```bash
cmake -S gateway_cpp -B gateway_cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build gateway_cpp/build -j8
```

产物：`gateway_cpp/build/tybok_gateway_cpp`。

注意事项：

- `image.cpp` 必须带 `-mavx2 -mfma -ffp-contract=fast`（CMakeLists 已对 `src/image.cpp`
  单独设置）——resize 与 torch `F.interpolate(bilinear, align_corners=False)` + `F.pad`
  **逐位一致**的编译前提，去掉会漂移 1 ulp。
- 只依赖 CUDA **driver** API（`cuMemAlloc`/`cuMemcpyHtoDAsync`/`cuIpcGetMemHandle`/`cuEvent`），
  不链接 torch。

## 运行

```bash
# 1) 先起推理 worker（Python）
python -m tybok worker --model <ckpt> --graph --socket /tmp/tybok_worker.sock

# 2) 起 C++ 网关
./gateway_cpp/build/tybok_gateway_cpp --worker-socket /tmp/tybok_worker.sock --port 8765 [--gpu-direct]
```

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--worker-socket PATH` | `/tmp/tybok_worker.sock` | worker Unix socket |
| `--host HOST` | `0.0.0.0` | 绑定地址 |
| `--port PORT` | `8765` | WebSocket 端口 |
| `--gpu-direct` | 关闭 | 开启 GPU 直传（HtoD + cudaIpcMemHandle 零拷贝 IPC）；默认走字节传输 |
| `--gpu-device N` | `0` | GPU 直传使用的设备（仅 `--gpu-direct` 时生效） |
| `--threads N` | 硬件线程数 | 解码/resize 线程池大小 |
| `--timing` | 关闭 | 打印每请求阶段耗时（stderr） |

## 两种传输（与 Python 网关对齐，GPU 直连是可选项）

- **字节传输（默认）**：与 Python 网关 `encode_request` 同款的 inline-payload 帧
  （`[4B total][4B hlen][JSON header][float32 图像/state 载荷]`），worker 端 `np.frombuffer`
  零拷贝视图。不初始化 CUDA，无 GPU 环境可直接运行。
- **GPU 直传（`--gpu-direct`）**：帧内不携带 tensor 字节，每个 tensor spec 带 base64 的
  `cudaIpcMemHandle` 导入元数据；worker 用 `torch.UntypedStorage._new_shared_cuda` 零拷贝
  导入。GPU 初始化失败时**直接退出**（不静默降级）。

两种传输模型看到的像素逐位一致，与 Python 网关两种传输也逐位一致。

## 验证

```bash
# resize 与 torch 逐位一致（CMake 目标 test_image）：读原始 RGB，写 float32 CHW
gateway_cpp/build/test_image IN.rgb H W TARGET_W TARGET_H OUT.f32
```

## 已知注意事项

- 多进程测试前先 `pkill -f 'tybok worker'` / `pkill -f tybok_gateway_cpp` 并删除
  `/tmp/vla_*.sock*`，否则端口/socket 冲突。
- GPU 直传的 source tensor 生命周期覆盖整个 worker roundtrip（`finalize` 里
  `unique_ptr<Request>` 函数作用域持有，roundtrip 后析构）——不要把它挪进 `if` 块作用域，
  否则 worker 打开已释放的内存（`CUDA error: invalid argument`）。
