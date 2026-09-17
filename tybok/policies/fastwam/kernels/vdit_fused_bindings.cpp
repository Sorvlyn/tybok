// vdit_fused_bindings.cpp -- pybind: single-kernel fusion of the video DiT block tail (cooperative)
//
// A separate extension fastwam_video_fused_ext (compiled apart from the action's fastwam_action_fused_ext),
// built on demand only under --cu-fused-vdit.
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void vdit_ffn_cuda(const void* x_in, const void* gate_mlp,
                                    const void* shift_mlp, const void* scale_mlp,
                                    const void* w0, const float* sw0, const void* b0,
                                    const void* w1, const float* sw1, const void* b1,
                                    void* out, void* a8buf, float* sa0buf,
                                    void* gbuf, void* a8gbuf, float* sa1buf,
                                    uint32_t* raw1, int stop_phase, int only_phase,
                                    int use_norm, float norm_eps, int mod_srow,
                                    unsigned long long* phase_ts, cudaStream_t stream);

extern "C" void vf_cast_gbuf_cuda(const void* g, const float* sa1, void* a8g,
                                  int Mn, int Fn, cudaStream_t stream);

// Device-capacity queries: the cooperative kernels' grid is now taken from `occupancy × SM count` (no longer hard-coded to 132).
// The phase_ts mark buffer spans gridDim.x, and the caller (the timing script) needs the actual value.
extern "C" int vdit_attn_self_grid();
extern "C" int vdit_attn_cross_grid();
extern "C" int vdit_ffn_grid();
// The grid of the non-cooperative split-phase form (no co-residency required; lower bound = VA_M / VG_M / VF_M, a plain launch may take multiple waves)
extern "C" int vdit_attn_self_split_grid();
extern "C" int vdit_attn_cross_split_grid();
extern "C" int vdit_ffn_split_grid();

static void vf_cast_gbuf_wrap(int64_t g, int64_t sa1, int64_t a8g, int64_t Mn,
                              int64_t Fn, int64_t stream) {
    vf_cast_gbuf_cuda(reinterpret_cast<const void*>(g),
                      reinterpret_cast<const float*>(sa1),
                      reinterpret_cast<void*>(a8g), (int)Mn, (int)Fn,
                      reinterpret_cast<cudaStream_t>(stream));
}

static void ffn_vdit_fused_wrap(int64_t x_in, int64_t gate_mlp,
                                int64_t shift_mlp, int64_t scale_mlp,
                                int64_t w0, int64_t sw0, int64_t b0,
                                int64_t w1, int64_t sw1, int64_t b1,
                                int64_t out, int64_t a8buf, int64_t sa0buf,
                                int64_t gbuf, int64_t a8gbuf, int64_t sa1buf,
                                int64_t raw1, int64_t stop_phase, int64_t only_phase,
                                int64_t use_norm, double norm_eps, int64_t phase_ts,
                                int64_t stream, int64_t mod_srow) {
    vdit_ffn_cuda(
        reinterpret_cast<const void*>(x_in), reinterpret_cast<const void*>(gate_mlp),
        reinterpret_cast<const void*>(shift_mlp),
        reinterpret_cast<const void*>(scale_mlp),
        reinterpret_cast<const void*>(w0), reinterpret_cast<const float*>(sw0),
        reinterpret_cast<const void*>(b0),
        reinterpret_cast<const void*>(w1), reinterpret_cast<const float*>(sw1),
        reinterpret_cast<const void*>(b1),
        reinterpret_cast<void*>(out), reinterpret_cast<void*>(a8buf),
        reinterpret_cast<float*>(sa0buf), reinterpret_cast<void*>(gbuf),
        reinterpret_cast<void*>(a8gbuf), reinterpret_cast<float*>(sa1buf),
        reinterpret_cast<uint32_t*>(raw1), (int)stop_phase, (int)only_phase,
        (int)use_norm, (float)norm_eps, (int)mod_srow,
        reinterpret_cast<unsigned long long*>(phase_ts),
        reinterpret_cast<cudaStream_t>(stream));
}

extern "C" void vdit_attn_self_cuda(
    const void* x_in, const void* shift_msa, const void* scale_msa, const void* gate_msa,
    const void* wqkv, const float* swqkv, const void* bqkv,
    const void* wnq, const void* wnk,
    const void* cos_t, const void* sin_t,
    const void* wo, const float* swo, const void* bo,
    void* out, void* qkv, void* qc, void* kc, void* vc, void* attn,
    void* a8x, float* sa0x, void* a8a, float* sa0a,
    float norm_eps, int mod_srow, int stop_phase, int only_phase,
    float* dbg, unsigned long long* phase_ts, cudaStream_t stream);

static void attn_self_wrap(
    int64_t x_in, int64_t shift_msa, int64_t scale_msa, int64_t gate_msa,
    int64_t wqkv, int64_t swqkv, int64_t bqkv,
    int64_t wnq, int64_t wnk, int64_t cos_t, int64_t sin_t,
    int64_t wo, int64_t swo, int64_t bo,
    int64_t out, int64_t qkv, int64_t qc, int64_t kc, int64_t vc, int64_t attn,
    int64_t a8x, int64_t sa0x, int64_t a8a, int64_t sa0a,
    double norm_eps, int64_t mod_srow, int64_t stop_phase, int64_t only_phase,
    int64_t dbg, int64_t phase_ts, int64_t stream)
{
    vdit_attn_self_cuda(
        reinterpret_cast<const void*>(x_in),
        reinterpret_cast<const void*>(shift_msa),
        reinterpret_cast<const void*>(scale_msa),
        reinterpret_cast<const void*>(gate_msa),
        reinterpret_cast<const void*>(wqkv), reinterpret_cast<const float*>(swqkv),
        reinterpret_cast<const void*>(bqkv),
        reinterpret_cast<const void*>(wnq), reinterpret_cast<const void*>(wnk),
        reinterpret_cast<const void*>(cos_t), reinterpret_cast<const void*>(sin_t),
        reinterpret_cast<const void*>(wo), reinterpret_cast<const float*>(swo),
        reinterpret_cast<const void*>(bo),
        reinterpret_cast<void*>(out), reinterpret_cast<void*>(qkv),
        reinterpret_cast<void*>(qc), reinterpret_cast<void*>(kc),
        reinterpret_cast<void*>(vc), reinterpret_cast<void*>(attn),
        reinterpret_cast<void*>(a8x), reinterpret_cast<float*>(sa0x),
        reinterpret_cast<void*>(a8a), reinterpret_cast<float*>(sa0a),
        (float)norm_eps, (int)mod_srow, (int)stop_phase, (int)only_phase,
        reinterpret_cast<float*>(dbg),
        reinterpret_cast<unsigned long long*>(phase_ts),
        reinterpret_cast<cudaStream_t>(stream));
}

extern "C" void vdit_attn_cross_cuda(
    const void* x_in, const void* context, const void* w3, const void* b3,
    const void* wq, const float* swq, const void* bq,
    const void* wkv, const float* swkv, const void* bkv,
    const void* wnq, const void* wnk,
    const void* wo, const float* swo, const void* bo, const void* ones,
    void* out, void* qp, void* kvp, void* qc, void* kc, void* vc, void* attn,
    void* a8q, float* sa0q, void* a8c, float* sa0c, void* a8o, float* sa0o,
    float norm_eps, int stop_phase, int only_phase, float* dbg,
    unsigned long long* phase_ts, cudaStream_t stream);

static void attn_cross_wrap(
    int64_t x_in, int64_t context, int64_t w3, int64_t b3,
    int64_t wq, int64_t swq, int64_t bq,
    int64_t wkv, int64_t swkv, int64_t bkv,
    int64_t wnq, int64_t wnk,
    int64_t wo, int64_t swo, int64_t bo, int64_t ones,
    int64_t out, int64_t qp, int64_t kvp, int64_t qc, int64_t kc, int64_t vc,
    int64_t attn, int64_t a8q, int64_t sa0q, int64_t a8c, int64_t sa0c,
    int64_t a8o, int64_t sa0o, double norm_eps, int64_t stop_phase, int64_t only_phase,
    int64_t dbg, int64_t phase_ts, int64_t stream)
{
    vdit_attn_cross_cuda(
        reinterpret_cast<const void*>(x_in), reinterpret_cast<const void*>(context),
        reinterpret_cast<const void*>(w3), reinterpret_cast<const void*>(b3),
        reinterpret_cast<const void*>(wq), reinterpret_cast<const float*>(swq),
        reinterpret_cast<const void*>(bq),
        reinterpret_cast<const void*>(wkv), reinterpret_cast<const float*>(swkv),
        reinterpret_cast<const void*>(bkv),
        reinterpret_cast<const void*>(wnq), reinterpret_cast<const void*>(wnk),
        reinterpret_cast<const void*>(wo), reinterpret_cast<const float*>(swo),
        reinterpret_cast<const void*>(bo), reinterpret_cast<const void*>(ones),
        reinterpret_cast<void*>(out), reinterpret_cast<void*>(qp),
        reinterpret_cast<void*>(kvp), reinterpret_cast<void*>(qc),
        reinterpret_cast<void*>(kc), reinterpret_cast<void*>(vc),
        reinterpret_cast<void*>(attn),
        reinterpret_cast<void*>(a8q), reinterpret_cast<float*>(sa0q),
        reinterpret_cast<void*>(a8c), reinterpret_cast<float*>(sa0c),
        reinterpret_cast<void*>(a8o), reinterpret_cast<float*>(sa0o),
        (float)norm_eps, (int)stop_phase, (int)only_phase, reinterpret_cast<float*>(dbg),
        reinterpret_cast<unsigned long long*>(phase_ts),
        reinterpret_cast<cudaStream_t>(stream));
}

// Geometry self-report (shape constants + GEMM template parameters)
extern "C" const char* vdit_attn_self_geom();
extern "C" const char* vdit_attn_cross_geom();
extern "C" const char* vdit_ffn_geom();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_self_grid", &vdit_attn_self_grid, "actual launch grid of the self kernel's cooperative form");
    m.def("attn_cross_grid", &vdit_attn_cross_grid, "actual launch grid of the cross kernel's cooperative form");
    m.def("ffn_grid", &vdit_ffn_grid, "actual launch grid of the video FFN kernel's cooperative form");
    m.def("attn_self_split_grid", &vdit_attn_self_split_grid, "actual launch grid of the self kernel's split-phase form");
    m.def("attn_cross_split_grid", &vdit_attn_cross_split_grid, "actual launch grid of the cross kernel's split-phase form");
    m.def("ffn_split_grid", &vdit_ffn_split_grid, "actual launch grid of the video FFN kernel's split-phase form");
    m.def("attn_self_geom", &vdit_attn_self_geom, "geometry self-report of the video self kernel");
    m.def("attn_cross_geom", &vdit_attn_cross_geom, "geometry self-report of the video cross kernel");
    m.def("ffn_geom", &vdit_ffn_geom, "geometry self-report of the video FFN kernel");
    m.def("vdit_attn_self", &attn_self_wrap,
          "video self-attention single-kernel fusion: norm1+modulate+quantization → qkv GEMM → "
          "qk-norm+3D-RoPE → bf16 flash → quantization → o GEMM + gate_mlp residual",
          py::arg("x_in"), py::arg("shift_msa"), py::arg("scale_msa"), py::arg("gate_msa"),
          py::arg("wqkv"), py::arg("swqkv"), py::arg("bqkv"),
          py::arg("wnq"), py::arg("wnk"), py::arg("cos_t"), py::arg("sin_t"),
          py::arg("wo"), py::arg("swo"), py::arg("bo"),
          py::arg("out"), py::arg("qkv"), py::arg("qc"), py::arg("kc"),
          py::arg("vc"), py::arg("attn"),
          py::arg("a8x"), py::arg("sa0x"), py::arg("a8a"), py::arg("sa0a"),
          py::arg("norm_eps"), py::arg("mod_srow"), py::arg("stop_phase"),
          py::arg("only_phase"), py::arg("dbg"), py::arg("phase_ts"), py::arg("stream"));
    m.def("vdit_attn_cross", &attn_cross_wrap,
          "video cross-attention single-kernel fusion: norm3+quantization → cross q/kv GEMM → "
          "qk-norm → bf16 flash → quantization → cross o GEMM + ungated residual",
          py::arg("x_in"), py::arg("context"), py::arg("w3"), py::arg("b3"),
          py::arg("wq"), py::arg("swq"), py::arg("bq"),
          py::arg("wkv"), py::arg("swkv"), py::arg("bkv"),
          py::arg("wnq"), py::arg("wnk"),
          py::arg("wo"), py::arg("swo"), py::arg("bo"), py::arg("ones"),
          py::arg("out"), py::arg("qp"), py::arg("kvp"), py::arg("qc"),
          py::arg("kc"), py::arg("vc"), py::arg("attn"),
          py::arg("a8q"), py::arg("sa0q"), py::arg("a8c"), py::arg("sa0c"),
          py::arg("a8o"), py::arg("sa0o"),
          py::arg("norm_eps"), py::arg("stop_phase"), py::arg("only_phase"), py::arg("dbg"),
          py::arg("phase_ts"), py::arg("stream"));

    m.def("cast_gbuf", &vf_cast_gbuf_wrap, "bf16 gbuf -> fp8 a8g (single pass, for the unfused baseline)",
          py::arg("g"), py::arg("sa1"), py::arg("a8g"), py::arg("M"), py::arg("N"),
          py::arg("stream"));
    m.def("vdit_ffn", &ffn_vdit_fused_wrap,
          "video DiT FFN single-kernel fusion: input quantization→up GEMM→GELU→quantization→down GEMM→gate_mlp residual",
          py::arg("x_in"), py::arg("gate_mlp"), py::arg("shift_mlp"), py::arg("scale_mlp"),
          py::arg("w0"), py::arg("sw0"), py::arg("b0"),
          py::arg("w1"), py::arg("sw1"), py::arg("b1"),
          py::arg("out"), py::arg("a8buf"), py::arg("sa0buf"),
          py::arg("gbuf"), py::arg("a8gbuf"), py::arg("sa1buf"),
          py::arg("raw1"), py::arg("stop_phase"), py::arg("only_phase"), py::arg("use_norm"),
          py::arg("norm_eps"), py::arg("phase_ts"), py::arg("stream"),
          py::arg("mod_srow"));
}
