// action_pre_bindings.cpp -- pybind for action_pre.cu (action DiT pre-component fused kernels)
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void ap_time_path_cuda(const float* ts, const double* freq,
                                  const unsigned char* w0t, const float* sw0,
                                  const void* b0,
                                  const unsigned char* w1t, const float* sw1,
                                  const void* b1,
                                  const unsigned char* w2t, const float* sw2,
                                  const void* b2,
                                  void* y0, void* y1, void* y2,
                                  float* amax0, float* amax1, float* amax2,
                                  void* tmod, cudaStream_t stream);

extern "C" void ap_ctx_repack_cuda(const void* out_big, int L, int NB,
                                   const void* wnk, float eps, void* kv_out,
                                   cudaStream_t stream);

static void ap_time_path_wrap(int64_t ts, int64_t freq,
                              int64_t w0t, int64_t sw0, int64_t b0,
                              int64_t w1t, int64_t sw1, int64_t b1,
                              int64_t w2t, int64_t sw2, int64_t b2,
                              int64_t y0, int64_t y1, int64_t y2,
                              int64_t amax0, int64_t amax1, int64_t amax2,
                              int64_t tmod, int64_t stream) {
    ap_time_path_cuda(
        reinterpret_cast<const float*>(ts), reinterpret_cast<const double*>(freq),
        reinterpret_cast<const unsigned char*>(w0t), reinterpret_cast<const float*>(sw0),
        reinterpret_cast<const void*>(b0),
        reinterpret_cast<const unsigned char*>(w1t), reinterpret_cast<const float*>(sw1),
        reinterpret_cast<const void*>(b1),
        reinterpret_cast<const unsigned char*>(w2t), reinterpret_cast<const float*>(sw2),
        reinterpret_cast<const void*>(b2),
        reinterpret_cast<void*>(y0), reinterpret_cast<void*>(y1), reinterpret_cast<void*>(y2),
        reinterpret_cast<float*>(amax0), reinterpret_cast<float*>(amax1),
        reinterpret_cast<float*>(amax2),
        reinterpret_cast<void*>(tmod), reinterpret_cast<cudaStream_t>(stream));
}

static void ap_ctx_repack_wrap(int64_t out_big, int64_t L, int64_t NB,
                               int64_t wnk, double eps, int64_t kv_out,
                               int64_t stream) {
    ap_ctx_repack_cuda(reinterpret_cast<const void*>(out_big), (int)L, (int)NB,
                       reinterpret_cast<const void*>(wnk), (float)eps,
                       reinterpret_cast<void*>(kv_out),
                       reinterpret_cast<cudaStream_t>(stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ap_time_path", &ap_time_path_wrap,
          "fused action time path: sinusoidal → FC0+SiLU → FC1+SiLU → FC2",
          py::arg("ts"), py::arg("freq"),
          py::arg("w0t"), py::arg("sw0"), py::arg("b0"),
          py::arg("w1t"), py::arg("sw1"), py::arg("b1"),
          py::arg("w2t"), py::arg("sw2"), py::arg("b2"),
          py::arg("y0"), py::arg("y1"), py::arg("y2"),
          py::arg("amax0"), py::arg("amax1"), py::arg("amax2"),
          py::arg("tmod"), py::arg("stream"));
    m.def("ap_ctx_repack", &ap_ctx_repack_wrap,
          "context kv repack + norm_k: [L, NB*6144] -> [NB,2,L,3072]",
          py::arg("out_big"), py::arg("L"), py::arg("NB"),
          py::arg("wnk"), py::arg("eps"), py::arg("kv_out"), py::arg("stream"));
}
