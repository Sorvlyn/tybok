// grid / capacity queries for the fused kernels: computes occupancy x SM for the **current device** instead of a hard-coded SM count.
//
// Two launch entry points:
//   occ_x_sm()  computes occupancy x SM only (no co-residency required), for the non-cooperative split-phase form;
//   coop_grid() validates the structural lower bound min_grid on top of occ_x_sm(): grid < min_grid **undercounts**
//               (it does not merely run slower); it throws when the capacity is insufficient.
// On failure it throws std::runtime_error (turned into RuntimeError by pybind), never exit(1) --
// exit bypasses Python and kills the process directly.
// The FASTWAM_COOP_GRID environment variable can force the cooperative grid at runtime (it must lie in
// [structural lower bound, device capacity]; out of bounds raises an error instead of silently clamping).
// cached_coop_grid / cached_split_grid: each kernel caches the query once in a static slot (initial slot value -1).
#ifndef FASTWAM_KERNELS_COOP_GRID_H_
#define FASTWAM_KERNELS_COOP_GRID_H_

#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ------------------------------------------------------------------ //
// Failure: throw an exception, do not exit(1)
// ------------------------------------------------------------------ //
// These functions are compiled into the extension via pybind11; `exit(1)` bypasses Python and
// terminates the process, whereas `throw` becomes `RuntimeError` so the caller can fall back, retry or report an error.
[[noreturn]] static inline void coop_fail(const char* name, const std::string& msg) {
    throw std::runtime_error(std::string(name) + ": " + msg);
}

// Current device. **Do not hard-code 0**: on a multi-GPU machine with `--device cuda:N`, the latter is the one doing the work.
static inline int coop_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess) return 0;
    return d;
}

// ------------------------------------------------------------------ //
// Pure query: does not launch, does not throw, writes nothing to stderr (used by registry / probe)
// ------------------------------------------------------------------ //
struct CudaPlan {
    int capacity;    // occupancy x SM (the cooperative launch upper bound; 0 = could not be queried)
    int occ;         // CTA/SM (1 = single-wave co-residency only; <2 usually means dynamic smem lost an 8KB granule)
    int sms;         // number of SMs on the current device
    int min_grid;    // structural lower bound declared by the caller
    int coop_ok;     // whether the device supports cooperative launch
    int query_ok;    // whether the query itself succeeded (distinguishes "API error" from "device does not support cooperative")
    int ok;          // query_ok && capacity is sufficient && coop_ok
};

// NOTE: so that the occupancy query reflects the real usage, this sets MaxDynamicSharedMemorySize
// once via `cudaFuncSetAttribute` (idempotent, and required before launch anyway). No other side effects.
static inline CudaPlan cuda_plan_query(const void* kernel, int nt, int smem, int min_grid) {
    CudaPlan p{0, 0, 0, min_grid, 0, 0, 0};
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
        != cudaSuccess) {
        return p;                      // query_ok = 0: dynamic smem exceeds this device's limit
    }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&p.occ, kernel, nt, smem) != cudaSuccess) {
        p.occ = 0;
        return p;
    }
    cudaDeviceProp prop;
    const int dev = coop_device();
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return p;
    p.query_ok = 1;
    p.coop_ok = prop.cooperativeLaunch ? 1 : 0;
    p.sms = prop.multiProcessorCount;
    p.capacity = p.occ * p.sms;
    p.ok = (p.capacity >= min_grid && p.coop_ok) ? 1 : 0;
    return p;
}

// ------------------------------------------------------------------ //
// Launch path: raises when the query fails or the grid does not fit
// ------------------------------------------------------------------ //
// kernel: kernel function pointer; nt: number of threads; smem: dynamic smem bytes (pass 0 for kernels with static smem).
// Returns `occupancy x SM`. Does not require co-residency (used by the plain launch path).
static inline int occ_x_sm(const void* kernel, int nt, int smem, const char* name) {
    // Dynamic smem must land exactly on the 8KB granule, otherwise 2 CTA/SM silently drops to 1 (the compiler reports nothing).
    const CudaPlan p = cuda_plan_query(kernel, nt, smem, 1);
    if (p.capacity <= 0) {
        coop_fail(name, "failed to query occupancy x SM (cudaFuncSetAttribute or "
                        "cudaOccupancyMaxActiveBlocksPerMultiprocessor returned an error)");
    }
    // NOTE: occ == 1 is not an error -- the split-phase path is equally correct across multiple waves, and
    // some kernels are **designed** for 1 CTA/SM (the tmt5_* ones with 96KB of dynamic smem). Callers that need occ
    // should read it from `cuda_plan_query()`; do not log here -- flushing stdout on a normal configuration is taken as a fault.
    return p.capacity;
}

// Cooperative path: device capacity + structural lower bound validation (plus an optional runtime override).
// min_grid: the grid must be >= it, otherwise phases are undercounted; name: used in error messages.
static inline int coop_grid(const void* kernel, int nt, int smem, int min_grid,
                            const char* name) {
    const CudaPlan p = cuda_plan_query(kernel, nt, smem, min_grid);
    if (!p.query_ok) {
        // Reported separately from "device does not support cooperative": most likely the dynamic smem exceeds this device's opt-in limit.
        coop_fail(name, "capability query failed: cudaFuncSetAttribute(MaxDynamicSharedMemorySize="
                        + std::to_string(smem) + ") or the occupancy query returned an error"
                        " (does the dynamic smem exceed this device's limit?)");
    }
    if (!p.coop_ok) {
        coop_fail(name, "this device does not support cooperative launch (use the non-cooperative split-phase form, or a different device / partition)");
    }
    if (p.capacity < min_grid) {
        coop_fail(name,
                  "device capacity " + std::to_string(p.capacity) + " CTA (occupancy "
                  + std::to_string(p.occ) + " × " + std::to_string(p.sms) + " SM) "
                  + "< structural lower bound " + std::to_string(min_grid)
                  + ". This cooperative kernel requires the whole grid to be co-resident at once, so this device /"
                    " partition cannot run this fused kernel (switch to the non-cooperative split-phase form, or to a larger device / partition).");
    }
    // For on-the-fly tuning / validation: FASTWAM_COOP_GRID forces the grid, which must lie in [min_grid, capacity].
    // Use it to shrink the grid while co-existing with other tasks, or to validate the numerics / correctness of a smaller grid.
    // NOTE: this only affects the **cooperative** path; the non-cooperative fallback is unaffected. Out of bounds raises
    // an error rather than silently clamping -- this is an explicit override, so an illegal value is misuse and the caller (Python) gets a RuntimeError.
    if (const char* env = getenv("FASTWAM_COOP_GRID"); env && *env) {
        const int want = atoi(env);
        if (want > p.capacity) {
            coop_fail(name, "FASTWAM_COOP_GRID=" + std::to_string(want)
                            + " > device capacity " + std::to_string(p.capacity)
                            + " CTA (occupancy " + std::to_string(p.occ) + " × "
                            + std::to_string(p.sms) + " SM) -- pass an explicit value that does not exceed the capacity.");
        }
        if (want < min_grid) {
            coop_fail(name, "FASTWAM_COOP_GRID=" + std::to_string(want)
                            + " < structural lower bound " + std::to_string(min_grid));
        }
        return want;
    }
    return p.capacity;
}

// ------------------------------------------------------------------ //
// Cache wrappers (used by each kernel): these queries need occupancy + device properties every time, and their
// result never changes within a process, so each kernel caches it once in a static slot (**the initial value must be -1**); factored out to remove five verbatim-identical hand-written caches.
// ------------------------------------------------------------------ //
static inline int cached_coop_grid(int& slot, const void* kernel, int nt, int smem,
                                   int min_grid, const char* name) {
    if (slot < 0) slot = coop_grid(kernel, nt, smem, min_grid, name);
    return slot;
}

// Non-cooperative split-phase grid: max of the capacity and the structural lower bound (a plain launch may run in multiple waves, so small cards work too).
static inline int cached_split_grid(int& slot, const void* kernel, int nt, int smem,
                                    int min_grid, const char* name) {
    if (slot < 0) {
        const int g = occ_x_sm(kernel, nt, smem, name);
        slot = g < min_grid ? min_grid : g;
    }
    return slot;
}

#endif  // FASTWAM_KERNELS_COOP_GRID_H_
