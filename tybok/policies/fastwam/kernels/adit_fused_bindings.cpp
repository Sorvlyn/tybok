// pybind entry point for the action-expert fused kernels (S / G / FFN).
//   adit_attn_self  : the whole self-attn chain
//   adit_attn_cross : the whole cross-attn path
//   adit_ffn        : FFN
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>

// ---- Actual launch grid / cooperative availability queries for registry/probe ----
// All three kernels' grids are set by the **problem size** (S=96 / G=48 / P3=GRID0×B); the
// cooperative form only additionally requires the whole grid to be co-resident. When it does not
// fit, *_coop_grid raises std::runtime_error (pybind turns that into RuntimeError).
extern "C" int adit_attn_self_grid();
extern "C" int adit_attn_self_coop_grid();
extern "C" int adit_attn_cross_grid();
extern "C" int adit_attn_cross_coop_grid(int L);
extern "C" int adit_ffn_grid(int B);
extern "C" int adit_ffn_coop_grid(int B);

// ---- S: adit_attn_self.cu ----
extern "C" void adit_attn_self_cuda(
    const void* x_in, const void* shift_msa, const void* scale_msa,
    const void* wqkv, const float* swqkv, const void* bqkv,
    const void* kv_cache, const void* v_cache,
    const void* wnq, const void* wnk,
    const void* cos_t, const void* sin_t,
    const void* gate_msa, const void* wo, const float* swo, const void* bo,
    void* out, void* qkv, void* attn,
    void* a8x, float* sa0x, void* a8a, float* sa0a, float* rstd_scratch,
    int L, int only_phase, cudaStream_t stream);

static void fused_self_g_wrap(
    const int64_t x_in, const int64_t shift_msa, const int64_t scale_msa,
    const int64_t wqkv, const int64_t swqkv, const int64_t bqkv,
    const int64_t kv_cache, const int64_t v_cache,
    const int64_t wnq, const int64_t wnk,
    const int64_t cos_t, const int64_t sin_t,
    const int64_t gate_msa, const int64_t wo, const int64_t swo, const int64_t bo,
    const int64_t out, const int64_t qkv, const int64_t attn,
    const int64_t a8x, const int64_t sa0x, const int64_t a8a, const int64_t sa0a,
    const int64_t rstd_scratch, const int64_t L, const int64_t stream,
    const int64_t only_phase = -1) {
    adit_attn_self_cuda(
        reinterpret_cast<const void*>(x_in),
        reinterpret_cast<const void*>(shift_msa),
        reinterpret_cast<const void*>(scale_msa),
        reinterpret_cast<const void*>(wqkv),
        reinterpret_cast<const float*>(swqkv),
        reinterpret_cast<const void*>(bqkv),
        reinterpret_cast<const void*>(kv_cache),
        reinterpret_cast<const void*>(v_cache),
        reinterpret_cast<const void*>(wnq),
        reinterpret_cast<const void*>(wnk),
        reinterpret_cast<const void*>(cos_t),
        reinterpret_cast<const void*>(sin_t),
        reinterpret_cast<const void*>(gate_msa),
        reinterpret_cast<const void*>(wo),
        reinterpret_cast<const float*>(swo),
        reinterpret_cast<const void*>(bo),
        reinterpret_cast<void*>(out),
        reinterpret_cast<void*>(qkv),
        reinterpret_cast<void*>(attn),
        reinterpret_cast<void*>(a8x),
        reinterpret_cast<float*>(sa0x),
        reinterpret_cast<void*>(a8a),
        reinterpret_cast<float*>(sa0a),
        reinterpret_cast<float*>(rstd_scratch),
        static_cast<int>(L),
        static_cast<int>(only_phase),
        reinterpret_cast<cudaStream_t>(stream));
}

// ---- G: adit_attn_cross.cu ----
extern "C" void adit_attn_cross_cuda(
    const void* x_in, const void* w3, const void* b3,
    const void* wq, const float* swq, const void* bq,
    void* qp, void* a8q, float* sa0q,
    const void* wnq, const void* k_cache, const void* v_cache, const void* mask,
    void* attn,
    const void* wo, const float* swo, const void* bo,
    void* out, void* a8o, float* sa0o, float* rstd_scratch,
    int L, int only_phase, cudaStream_t stream);

static void cross_g_wrap(
    const int64_t x_in, const int64_t w3, const int64_t b3,
    const int64_t wq, const int64_t swq, const int64_t bq,
    const int64_t qp, const int64_t a8q, const int64_t sa0q,
    const int64_t wnq, const int64_t k_cache, const int64_t v_cache, const int64_t mask,
    const int64_t attn,
    const int64_t wo, const int64_t swo, const int64_t bo,
    const int64_t out, const int64_t a8o, const int64_t sa0o, const int64_t rstd_scratch,
    const int64_t L, const int64_t stream, const int64_t only_phase = -1) {
    adit_attn_cross_cuda(
        reinterpret_cast<const void*>(x_in),
        reinterpret_cast<const void*>(w3),
        reinterpret_cast<const void*>(b3),
        reinterpret_cast<const void*>(wq),
        reinterpret_cast<const float*>(swq),
        reinterpret_cast<const void*>(bq),
        reinterpret_cast<void*>(qp),
        reinterpret_cast<void*>(a8q),
        reinterpret_cast<float*>(sa0q),
        reinterpret_cast<const void*>(wnq),
        reinterpret_cast<const void*>(k_cache),
        reinterpret_cast<const void*>(v_cache),
        reinterpret_cast<const void*>(mask),
        reinterpret_cast<void*>(attn),
        reinterpret_cast<const void*>(wo),
        reinterpret_cast<const float*>(swo),
        reinterpret_cast<const void*>(bo),
        reinterpret_cast<void*>(out),
        reinterpret_cast<void*>(a8o),
        reinterpret_cast<float*>(sa0o),
        reinterpret_cast<float*>(rstd_scratch),
        static_cast<int>(L),
        static_cast<int>(only_phase),
        reinterpret_cast<cudaStream_t>(stream));
}

// ---- FFN: adit_ffn.cu ----
extern "C" void adit_ffn_cuda(const void* x_in, const void* shift_mlp,
                                  const void* scale_mlp, const void* gate_mlp,
                                  const void* w0, const float* sw0, const void* b0,
                                  const void* w1, const float* sw1, const void* b1,
                                  void* out, void* a8buf, float* sa0buf,
                                  void* gbuf, void* a8gbuf, uint32_t* raw1,
                                  int B, int only_phase, cudaStream_t stream);

static void ffn_fused_p3_wrap(const int64_t x_in, const int64_t shift_mlp,
                              const int64_t scale_mlp, const int64_t gate_mlp,
                              const int64_t w0, const int64_t sw0, const int64_t b0,
                              const int64_t w1, const int64_t sw1, const int64_t b1,
                              const int64_t out, const int64_t a8buf, const int64_t sa0buf,
                              const int64_t gbuf, const int64_t a8gbuf, const int64_t raw1,
                              const int64_t B, const int64_t stream,
                              const int64_t only_phase = -1) {
    adit_ffn_cuda(reinterpret_cast<const void*>(x_in),
                      reinterpret_cast<const void*>(shift_mlp),
                      reinterpret_cast<const void*>(scale_mlp),
                      reinterpret_cast<const void*>(gate_mlp),
                      reinterpret_cast<const void*>(w0),
                      reinterpret_cast<const float*>(sw0),
                      reinterpret_cast<const void*>(b0),
                      reinterpret_cast<const void*>(w1),
                      reinterpret_cast<const float*>(sw1),
                      reinterpret_cast<const void*>(b1),
                      reinterpret_cast<void*>(out),
                      reinterpret_cast<void*>(a8buf),
                      reinterpret_cast<float*>(sa0buf),
                      reinterpret_cast<void*>(gbuf),
                      reinterpret_cast<void*>(a8gbuf),
                      reinterpret_cast<uint32_t*>(raw1),
                      (int)B,
                      static_cast<int>(only_phase),
                      reinterpret_cast<cudaStream_t>(stream));
}

// Geometry self-report (shape constants + GEMM template parameters)
extern "C" const char* adit_attn_self_geom();
extern "C" const char* adit_attn_cross_geom();
extern "C" const char* adit_ffn_geom();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adit_attn_self", &fused_self_g_wrap,
          "S: A(norm1+mod+qkv)+B+C(RMS/RoPE/flash)+D(o+gate residual) in one cooperative kernel",
          py::arg("x_in"), py::arg("shift_msa"), py::arg("scale_msa"),
          py::arg("wqkv"), py::arg("swqkv"), py::arg("bqkv"),
          py::arg("kv_cache"), py::arg("v_cache"),
          py::arg("wnq"), py::arg("wnk"),
          py::arg("cos_t"), py::arg("sin_t"),
          py::arg("gate_msa"), py::arg("wo"), py::arg("swo"), py::arg("bo"),
          py::arg("out"), py::arg("qkv"), py::arg("attn"),
          py::arg("a8x"), py::arg("sa0x"), py::arg("a8a"), py::arg("sa0a"),
          py::arg("rstd_scratch"), py::arg("L"), py::arg("stream"),
          py::arg("only_phase") = -1);
    m.def("adit_attn_cross", &cross_g_wrap,
          "G: E1(norm3+q')+E2(RMS+masked flash)+E3(o+residual) in one cooperative kernel",
          py::arg("x_in"), py::arg("w3"), py::arg("b3"),
          py::arg("wq"), py::arg("swq"), py::arg("bq"),
          py::arg("qp"), py::arg("a8q"), py::arg("sa0q"),
          py::arg("wnq"), py::arg("k_cache"), py::arg("v_cache"), py::arg("mask"),
          py::arg("attn"),
          py::arg("wo"), py::arg("swo"), py::arg("bo"),
          py::arg("out"), py::arg("a8o"), py::arg("sa0o"), py::arg("rstd_scratch"),
          py::arg("L"), py::arg("stream"), py::arg("only_phase") = -1);
    m.def("adit_ffn", &ffn_fused_p3_wrap, "P3 fused action-FFN (norm2+up+gelu+down+gate)",
          py::arg("x_in"), py::arg("shift_mlp"), py::arg("scale_mlp"), py::arg("gate_mlp"),
          py::arg("w0"), py::arg("sw0"), py::arg("b0"),
          py::arg("w1"), py::arg("sw1"), py::arg("b1"),
          py::arg("out"), py::arg("a8buf"), py::arg("sa0buf"), py::arg("gbuf"),
          py::arg("a8gbuf"), py::arg("raw1"), py::arg("B"), py::arg("stream"),
          py::arg("only_phase") = -1);
    // registry/probe: actual grid + cooperative availability (the latter raises RuntimeError when it does not fit)
    m.def("attn_self_grid", &adit_attn_self_grid, "S kernel's actual launch grid (problem size)");
    m.def("attn_self_coop_grid", &adit_attn_self_coop_grid, "S kernel cooperative-form check; returns the grid");
    m.def("adit_attn_cross_grid", &adit_attn_cross_grid, "G kernel's actual launch grid (problem size)");
    m.def("adit_attn_cross_coop_grid", &adit_attn_cross_coop_grid, py::arg("L"),
          "G kernel cooperative-form check; returns the grid");
    m.def("ffn_grid", &adit_ffn_grid, py::arg("B"),
          "P3 FFN kernel's actual launch grid (GRID0 × B)");
    m.def("ffn_coop_grid", &adit_ffn_coop_grid, py::arg("B"),
          "P3 FFN kernel cooperative-form check; returns the grid");
    m.def("attn_self_geom", &adit_attn_self_geom, "geometry self-report of the action S kernel");
    m.def("adit_attn_cross_geom", &adit_attn_cross_geom, "geometry self-report of the action G kernel");
    m.def("ffn_geom", &adit_ffn_geom, "geometry self-report of the action FFN kernel");
}
