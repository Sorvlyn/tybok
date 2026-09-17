// tmt5_fused_bindings.cpp -- pybind: single-kernel fusion of the UMT5 text encoder's two sublayers (cooperative + split-phase)
//
// A standalone extension fastwam_tmt5_fused_ext (compiled separately from the action / video extensions), built on
// demand only under --cu-fused-text-encoder. This file is the module entry point (an extension can only have one
// PYBIND11_MODULE); the attention sublayer's bindings live in tmt5_attn_bindings.cpp and are registered in by tmt5_attn_register().
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>

// tmt5_attn_bindings.cpp
void tmt5_attn_register(pybind11::module& m);

extern "C" void tmt5_ffn_cuda(const void* x_in, const void* nw,
                                  const void* w0, const float* sw0, const void* b0,
                                  const void* w1, const float* sw1, const void* b1,
                                  void* out, void* a8buf, float* sa0buf,
                                  void* gbuf, void* a8gbuf, float* sa1buf,
                                  int stop_phase, float norm_eps,
                                  unsigned long long* phase_ts, int only_phase,
                                  cudaStream_t stream);

// The actual launch grid (the phase_ts buffer span is gridDim.x, no longer a hard-coded 66)
extern "C" int tmt5_ffn_grid();
extern "C" int tmt5_ffn_coop_grid();

static void tmt5_ffn_fused_wrap(int64_t x_in, int64_t nw,
                              int64_t w0, int64_t sw0, int64_t b0,
                              int64_t w1, int64_t sw1, int64_t b1,
                              int64_t out, int64_t a8buf, int64_t sa0buf,
                              int64_t gbuf, int64_t a8gbuf, int64_t sa1buf,
                              int64_t stop_phase, double norm_eps,
                              int64_t phase_ts, int64_t stream,
                              int64_t only_phase = -1) {
    tmt5_ffn_cuda(
        reinterpret_cast<const void*>(x_in), reinterpret_cast<const void*>(nw),
        reinterpret_cast<const void*>(w0), reinterpret_cast<const float*>(sw0),
        reinterpret_cast<const void*>(b0),
        reinterpret_cast<const void*>(w1), reinterpret_cast<const float*>(sw1),
        reinterpret_cast<const void*>(b1),
        reinterpret_cast<void*>(out), reinterpret_cast<void*>(a8buf),
        reinterpret_cast<float*>(sa0buf), reinterpret_cast<void*>(gbuf),
        reinterpret_cast<void*>(a8gbuf), reinterpret_cast<float*>(sa1buf),
        (int)stop_phase, (float)norm_eps,
        reinterpret_cast<unsigned long long*>(phase_ts), (int)only_phase,
        reinterpret_cast<cudaStream_t>(stream));
}

// Geometry self-report (shape constants + GEMM template parameters)
extern "C" const char* tmt5_ffn_geom();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tmt5_ffn", &tmt5_ffn_fused_wrap,
          "UMT5 FFN sublayer single-kernel fusion (cooperative): S1 norm+quantization → S2 packed-wi GEMM → "
          "S3 gelu_new gating+quantization → S4 wo GEMM+residual",
          py::arg("x_in"), py::arg("nw"),
          py::arg("w0"), py::arg("sw0"), py::arg("b0"),
          py::arg("w1"), py::arg("sw1"), py::arg("b1"),
          py::arg("out"), py::arg("a8buf"), py::arg("sa0buf"),
          py::arg("gbuf"), py::arg("a8gbuf"), py::arg("sa1buf"),
          py::arg("stop_phase"), py::arg("norm_eps"),
          py::arg("phase_ts"), py::arg("stream"), py::arg("only_phase") = -1);
    m.def("ffn_grid", &tmt5_ffn_grid, "the FFN kernel's actual launch grid in its non-cooperative form");
    m.def("ffn_coop_grid", &tmt5_ffn_coop_grid, "the FFN kernel's actual launch grid in its cooperative form");
    m.def("ffn_geom", &tmt5_ffn_geom,
          "geometry self-report of the FFN kernel (shape constants + GEMM template parameters; the same macro set as the instantiation)");
    tmt5_attn_register(m);   // attention sublayer (tmt5_attn)
}
