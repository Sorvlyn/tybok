// Geometry self-report: reconciles the geometry table's declarations against the parameters actually compiled into the binary.
//
// The GEMM template parameters (BM/BN/BK/..., 29 in total) are factored into a macro; instantiation and self-report use **the same macro**:
//     fwam_fp8_gemm_body<FWAM_..._TILES>(...)   // instantiation
//     "F_PHASE_2=" FWAM_TILES(FWAM_..._TILES)  // self-report (stringifies the same token sequence)
// Therefore the self-report can never disagree with the instantiation -- this is what makes the geometry table trustworthy.
//
// Changing the geometry = edit the macro + recompile + run the check in `kernels/geometry.py`.
#ifndef FASTWAM_KERNELS_GEOM_REPORT_H_
#define FASTWAM_KERNELS_GEOM_REPORT_H_

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// Stringifies the macro arguments **after expansion** (same macro → same token sequence).
// ⚠️ Two levels are required: `#` suppresses argument expansion, so a direct `#__VA_ARGS__` prints only the macro name
// (observed in practice: it reported `F_PHASE_2=FWAM_TMT5_FFN_F_PHASE_2_TILES` instead of those 29 tokens).
#define FWAM_STR_(...)   #__VA_ARGS__
#define FWAM_STR(...)    FWAM_STR_(__VA_ARGS__)
#define FWAM_TILES(...)  FWAM_STR(__VA_ARGS__)

static inline std::string fwam_geom_join(const std::vector<std::string>& parts) {
    std::string out;
    for (const std::string& p : parts) {
        if (!out.empty()) out += " | ";
        out += p;
    }
    return out;
}

// Shape constants: k=v joined by commas (the values are integers)
static inline std::string fwam_shapes(
    const std::initializer_list<std::pair<const char*, long>>& kv) {
    std::string out;
    for (const auto& item : kv) {
        char b[64];
        snprintf(b, sizeof b, "%s=%ld", item.first, item.second);
        if (!out.empty()) out += ",";
        out += b;
    }
    return out;
}

// One template-parameter entry: <label>=<stringified macro>
static inline std::string fwam_tiles(const char* label, const char* tokens) {
    return std::string(label) + "=" + tokens;
}

#endif  // FASTWAM_KERNELS_GEOM_REPORT_H_
