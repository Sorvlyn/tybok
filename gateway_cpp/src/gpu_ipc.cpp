#include "gpu_ipc.h"

#include "hash.h"

namespace {
const char* cuda_err(CUresult r) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    return s ? s : "cuda error";
}
}  // namespace

bool GpuIpc::ctx_current() {
    return ctx_ && cuCtxSetCurrent(ctx_) == CUDA_SUCCESS;
}

GpuIpc::~GpuIpc() {
    if (!ctx_) return;
    cuCtxSetCurrent(ctx_);
    if (stream_) cuStreamDestroy(stream_);
    cuCtxDestroy(ctx_);
}

GpuIpc::Request::~Request() {
    if (!owner || !owner->ctx_current()) return;
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i]) cuMemFree(tensors[i]);
        if (events[i]) cuEventDestroy(events[i]);
    }
    if (ref_counter) cuMemFree(ref_counter);
}

bool GpuIpc::init(int device, std::string* err) {
    CUresult r;
    if ((r = cuInit(0)) != CUDA_SUCCESS) {
        if (err) *err = std::string("cuInit: ") + cuda_err(r);
        return false;
    }
    CUdevice dev;
    if ((r = cuDeviceGet(&dev, device)) != CUDA_SUCCESS) {
        if (err) *err = std::string("cuDeviceGet: ") + cuda_err(r);
        return false;
    }
#if CUDA_VERSION >= 13000
    // CUDA 13 replaced cuCtxCreate with the 4-arg v4 form (the extra CUctxCreateParams
    // carries exec-affinity / CIG options). A NULL params pointer means "regular context",
    // which is exactly what the pre-13 3-arg call created.
    r = cuCtxCreate(&ctx_, nullptr, CU_CTX_SCHED_AUTO, dev);
#else
    r = cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, dev);
#endif
    if (r != CUDA_SUCCESS) {
        if (err) *err = std::string("cuCtxCreate: ") + cuda_err(r);
        return false;
    }
    if ((r = cuStreamCreate(&stream_, CU_STREAM_DEFAULT)) != CUDA_SUCCESS) {
        if (err) *err = std::string("cuStreamCreate: ") + cuda_err(r);
        return false;
    }
    device_ = device;
    return true;
}

std::unique_ptr<GpuIpc::Request> GpuIpc::begin_request(std::string* err) {
    auto req = std::make_unique<Request>();
    req->owner = this;
    if (!ctx_current()) {
        if (err) *err = "no cuda context";
        return nullptr;
    }
    if (cuMemAlloc(&req->ref_counter, 64) != CUDA_SUCCESS) {
        if (err) *err = "cuMemAlloc(ref_counter) failed";
        return nullptr;
    }
    cuMemsetD8(req->ref_counter, 0, 64);
    return req;
}

bool GpuIpc::put_tensor(Request& req, const void* host, size_t bytes,
                        const std::vector<long long>& shape, long long ref_o,
                        GpuTensorMeta* meta, std::string* err) {
    if (!ctx_current()) {
        if (err) *err = "no cuda context";
        return false;
    }
    CUdeviceptr dev = 0;
    if (cuMemAlloc(&dev, bytes) != CUDA_SUCCESS) {
        if (err) *err = "cuMemAlloc failed";
        return false;
    }
    CUevent event = nullptr;
    if (cuMemcpyHtoDAsync(dev, host, bytes, stream_) != CUDA_SUCCESS) {
        if (err) *err = "cuMemcpyHtoDAsync failed";
        cuMemFree(dev);
        return false;
    }
    if (cuEventCreate(&event, CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS) != CUDA_SUCCESS) {
        if (err) *err = "cuEventCreate failed";
        cuMemFree(dev);
        return false;
    }
    if (cuEventRecord(event, stream_) != CUDA_SUCCESS) {
        if (err) *err = "cuEventRecord failed";
        cuMemFree(dev);
        cuEventDestroy(event);
        return false;
    }
    req.tensors.push_back(dev);
    req.events.push_back(event);

    CUipcMemHandle handle;
    CUipcMemHandle ref_handle;
    CUipcEventHandle ev_handle;
    if (cuIpcGetMemHandle(&handle, dev) != CUDA_SUCCESS) {
        if (err) *err = "cuIpcGetMemHandle failed";
        return false;
    }
    if (cuIpcGetMemHandle(&ref_handle, req.ref_counter) != CUDA_SUCCESS) {
        if (err) *err = "cuIpcGetMemHandle(ref) failed";
        return false;
    }
    if (cuIpcGetEventHandle(&ev_handle, event) != CUDA_SUCCESS) {
        if (err) *err = "cuIpcGetEventHandle failed";
        return false;
    }

    meta->handle_b64 = b64_encode(&handle, sizeof(handle));
    meta->ref_h_b64 = b64_encode(&ref_handle, sizeof(ref_handle));
    meta->ev_h_b64 = b64_encode(&ev_handle, sizeof(ev_handle));
    meta->ref_o = ref_o;
    meta->size_bytes = static_cast<long long>(bytes);
    meta->offset_bytes = 0;
    meta->device = device_;
    meta->shape = shape;
    return true;
}
