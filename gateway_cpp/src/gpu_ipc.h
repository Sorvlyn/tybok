// GPU-direct IPC via the CUDA *driver* API (public, no torch private API):
//
//   cuMemAlloc + cuMemcpyHtoDAsync  ->  gateway owns the frame on GPU
//   cuIpcGetMemHandle               ->  handle the worker imports zero-copy
//   cuEventCreate(INTERPROCESS) + cuEventRecord after the HtoD
//                                   ->  cross-process sync (the worker's
//                                       cudaStreamWaitEvent blocks until the
//                                       HtoD is visible)
//   one small ref-counter block per request (worker atomically +/-1 at ref_o
//   -- harmless bookkeeping)
//
// Lifetime: per-request. The gateway must hold the source allocations until
// the worker has imported them; the worker imports + D2D-copies during
// handle_frame and only then sends the response, so freeing a request's
// allocations right after its roundtrip completes is safe. This works with any
// number of concurrent clients (a fixed keep-alive ring would break once more
// requests are in flight than ring slots).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>

struct GpuTensorMeta {
    std::string handle_b64;  // cuIpcGetMemHandle of the tensor buffer
    std::string ref_h_b64;   // cuIpcGetMemHandle of the request ref-counter block
    long long ref_o = 0;     // index into the ref-counter block
    std::string ev_h_b64;    // cuIpcGetEventHandle
    long long size_bytes = 0;
    long long offset_bytes = 0;
    long long device = 0;
    std::vector<long long> shape;
};

class GpuIpc {
  public:
    GpuIpc() = default;
    ~GpuIpc();
    GpuIpc(const GpuIpc&) = delete;
    GpuIpc& operator=(const GpuIpc&) = delete;

    bool init(int device, std::string* err);

    struct Request {
        Request() = default;
        ~Request();
        Request(const Request&) = delete;
        Request& operator=(const Request&) = delete;

        GpuIpc* owner = nullptr;
        std::vector<CUdeviceptr> tensors;
        std::vector<CUevent> events;
        CUdeviceptr ref_counter = 0;
    };

    // Create the per-request ref-counter block; returns nullptr + *err on failure.
    std::unique_ptr<Request> begin_request(std::string* err);

    // Copy `bytes` of host data to device and fill *meta (incl. IPC handles).
    // Tensors of one request share the request's ref-counter block (ref_o =
    // tensor index).
    bool put_tensor(Request& req, const void* host, size_t bytes,
                    const std::vector<long long>& shape, long long ref_o,
                    GpuTensorMeta* meta, std::string* err);

  private:
    bool ctx_current();
    CUcontext ctx_ = nullptr;
    CUstream stream_ = nullptr;
    int device_ = 0;
};
