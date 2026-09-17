// tmt5_attn_bindings.cpp -- pybind registration: single-kernel fusion of the UMT5 text encoder's self-attention sublayer (cooperative + split-phase)
//
// Note: this file does **not** define PYBIND11_MODULE — an extension can only have one module entry point, and the
// entry point is in tmt5_fused_bindings.cpp (which calls tmt5_attn_register). The two sublayers share the
// single `fastwam_tmt5_fused_ext` extension.
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void tmt5_attn_cuda(const void* x_in, const void* nw,
                                   const void* wqkv, const float* sqkv, const void* bqkv,
                                   const void* wo, const float* swo, const void* bo,
                                   const void* posb, const void* amask,
                                   void* out, void* a8buf, float* sa0buf,
                                   void* qkvbuf, void* attnbuf, void* a8obuf, float* sa1buf,
                                   int stop_phase, float norm_eps,
                                   unsigned long long* phase_ts, float* dbg,
                                   int only_phase, cudaStream_t stream);

// The actual launch grid (the phase_ts buffer span is gridDim.x, no longer a hard-coded 66)
extern "C" int tmt5_attn_grid();
extern "C" int tmt5_attn_coop_grid();

static void tmt5_attn_fused_wrap(int64_t x_in, int64_t nw,
                               int64_t wqkv, int64_t sqkv, int64_t bqkv,
                               int64_t wo, int64_t swo, int64_t bo,
                               int64_t posb, int64_t amask,
                               int64_t out, int64_t a8buf, int64_t sa0buf,
                               int64_t qkvbuf, int64_t attnbuf, int64_t a8obuf, int64_t sa1buf,
                               int64_t stop_phase, double norm_eps,
                               int64_t phase_ts, int64_t dbg, int64_t stream,
                               int64_t only_phase = -1) {
    tmt5_attn_cuda(
        (const void*)x_in, (const void*)nw,
        (const void*)wqkv, (const float*)sqkv, (const void*)bqkv,
        (const void*)wo, (const float*)swo, (const void*)bo,
        (const void*)posb, (const void*)amask,
        (void*)out, (void*)a8buf, (float*)sa0buf,
        (void*)qkvbuf, (void*)attnbuf, (void*)a8obuf, (float*)sa1buf,
        (int)stop_phase, (float)norm_eps,
        (unsigned long long*)phase_ts, (float*)dbg, (int)only_phase,
        reinterpret_cast<cudaStream_t>(stream));
}

// Geometry self-report
extern "C" const char* tmt5_attn_geom();

void tmt5_attn_register(pybind11::module& m) {
    m.def("tmt5_attn", &tmt5_attn_fused_wrap,
          py::arg("x_in"), py::arg("nw"),
          py::arg("wqkv"), py::arg("sqkv"), py::arg("bqkv"),
          py::arg("wo"), py::arg("swo"), py::arg("bo"),
          py::arg("posb"), py::arg("amask"),
          py::arg("out"), py::arg("a8buf"), py::arg("sa0buf"),
          py::arg("qkvbuf"), py::arg("attnbuf"), py::arg("a8obuf"), py::arg("sa1buf"),
          py::arg("stop_phase"), py::arg("norm_eps"),
          py::arg("phase_ts"), py::arg("dbg") = 0, py::arg("stream"),
          py::arg("only_phase") = -1);
    m.def("attn_grid", &tmt5_attn_grid, "the attention kernel's actual launch grid in its non-cooperative form");
    m.def("attn_coop_grid", &tmt5_attn_coop_grid, "the attention kernel's actual launch grid in its cooperative form");
    m.def("attn_geom", &tmt5_attn_geom,
          "geometry self-report of the attention kernel (shape constants + GEMM template parameters; the same macro set as the instantiation)");
}
