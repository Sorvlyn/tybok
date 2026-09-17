"""Loading entry point for fastWAM fused CUDA kernels (build, cache, and error wrapping).

Each of the four extensions is compiled independently and built on demand: the action fused
kernels (S/G/FFN-P3) are always compiled; the video tail block (norm2+modulate+FFN+gate) only
under ``--cu-fused-vdit``; the action pre-components (time path + context kv) only under
``--action-pre-fused``; the UMT5 text encoder only under ``--cu-fused-text-encoder``.
Compilation goes through ``torch.utils.cpp_extension.load`` (cached by source content hash) and
requires a local nvcc + a CUDA toolkit matching torch; each loader raises a ``RuntimeError``
with guidance on failure.

Numerics contract: all activation I/O is **bf16**; weights fp8 e4m3fn + per-row fp32 scale +
bf16 bias (w8a8); quantization semantics amax/448 + software RNE, bf16 RN, bias added in fp32
after dequant -- same convention as the engine's ``FP8Linear``; S/G use bf16 flash (online
softmax) instead of fp32 SDPA (pure bf16 semantics tier). Fixed geometry (action expert):
M=32, hidden=1024, attn=3072 (24x128), ffn=4096.
"""

from __future__ import annotations

import functools
import os

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

_EXT_NAME = "fastwam_action_fused_ext"

# Single source of truth for the source-file list: shared by the loader and the sweep-variant
# build, so the two can't compile different kernels.
_ACTION_SOURCES = ("adit_attn_self.cu", "adit_attn_cross.cu", "adit_ffn.cu", "adit_fused_bindings.cpp")


def _srcs(names: tuple[str, ...]) -> list:
    return [os.path.join(_KERNEL_DIR, n) for n in names]


def build_variant(name: str, src_dir: str, sources: tuple[str, ...]) -> object:
    """Compile a variant extension from another source directory (for sweeps that build
    geometry variants).

    Uses the same flags as the production loader (``-O3``, and **no** ``--use_fast_math``), so
    the production extension is unaffected. ``name`` must be globally unique: sweeps name by
    source content hash, so the same geometry is compiled only once.
    """
    from torch.utils.cpp_extension import load

    return load(
        name=name,
        sources=[os.path.join(src_dir, n) for n in sources],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _build() -> object:
    from torch.utils.cpp_extension import load

    return load(
        name=_EXT_NAME,
        sources=_srcs(_ACTION_SOURCES),
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def load_fused_ext() -> object:
    """Compile and load the S/G/FFN fused-kernel extension (in-process singleton, cached by
    content hash).

    On build failure raises a ``RuntimeError`` with guidance (missing nvcc, CUDA toolkit /
    torch version mismatch, etc.).
    """
    try:
        return _build()
    except Exception as exc:  # noqa: BLE001 - wrap into an error with guidance
        raise RuntimeError(
            "failed to build the fastwam action-fused CUDA extension "
            f"('{_EXT_NAME}'); needs nvcc + a CUDA toolkit matching torch "
            f"(torch {_import_torch_version()}). "
            f"Original error: {exc}"
        ) from exc


_VIDEO_EXT_NAME = "fastwam_video_fused_ext"

_VIDEO_SOURCES = ("vdit_ffn.cu", "vdit_attn_self.cu", "vdit_attn_cross.cu", "vdit_fused_bindings.cpp")


def _build_video() -> object:
    from torch.utils.cpp_extension import load

    return load(
        name=_VIDEO_EXT_NAME,
        sources=_srcs(_VIDEO_SOURCES),
        extra_cuda_cflags=["-O3"],  # note: do **not** add --use_fast_math
        extra_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def load_video_fused_ext() -> object:
    """Compile and load the single-kernel fused extension for the video block tail
    (norm2+modulate+FFN+gate).

    Compiled separately from the action extension, built on demand only under
    ``--cu-fused-vdit``, so it does not add to the action path's startup time; numerics are the
    quantization-drift tier (relative error <=3e-2).
    """
    try:
        return _build_video()
    except Exception as exc:  # noqa: BLE001 - wrap into an error with guidance
        raise RuntimeError(
            "failed to build the fastwam video-fused CUDA extension "
            f"('{_VIDEO_EXT_NAME}'); needs nvcc + a CUDA toolkit matching torch "
            f"(torch {_import_torch_version()}). "
            f"Original error: {exc}"
        ) from exc


_ACTION_PRE_EXT_NAME = "fastwam_action_pre_ext"


def _build_action_pre() -> object:
    from torch.utils.cpp_extension import load

    return load(
        name=_ACTION_PRE_EXT_NAME,
        sources=[
            os.path.join(_KERNEL_DIR, "action_pre.cu"),
            os.path.join(_KERNEL_DIR, "action_pre_bindings.cpp"),
        ],
        extra_cuda_cflags=["-O3"],  # note: do **not** add --use_fast_math
        extra_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def load_action_pre_ext() -> object:
    """Compile and load the action DiT pre-component fused kernels (time path + context kv
    precompute).

    Compiled separately from the action/video/text extensions, built on demand only under
    ``--action-pre-fused``. Numerics: the context path is bitwise identical to production; the
    time path differs only in fp32 reduction order (sub-ulp in bf16).
    """
    try:
        return _build_action_pre()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "failed to build the fastwam action-pre-fused CUDA extension "
            f"('{_ACTION_PRE_EXT_NAME}'); needs nvcc + a CUDA toolkit matching torch "
            f"(torch {_import_torch_version()}). "
            f"Original error: {exc}"
        ) from exc


_TMT5_EXT_NAME = "fastwam_tmt5_fused_ext"

_TMT5_SOURCES = ("tmt5_ffn.cu", "tmt5_attn.cu", "tmt5_fused_bindings.cpp", "tmt5_attn_bindings.cpp")


def _build_tmt5() -> object:
    from torch.utils.cpp_extension import load

    return load(
        name=_TMT5_EXT_NAME,
        sources=_srcs(_TMT5_SOURCES),
        extra_cuda_cflags=["-O3"],  # note: do **not** add --use_fast_math
        extra_cflags=["-O3"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def load_tmt5_fused_ext() -> object:
    """Compile and load the single-kernel fused extension for the UMT5 text encoder's two
    sublayers (in-process singleton).

    Compiled separately from the action/video extensions, built on demand only under
    ``--cu-fused-text-encoder``, so it does not add to the startup time of other paths.
    Numerics contract: **bitwise-aligned with the existing Triton chain** (the FFN's
    F_PHASE_2/3/4 and attention's S_PHASE_2~5 are bitwise identical), except that S_PHASE_1's
    row reduction order differs from Triton's ``tl.sum`` tree (~0.0005% of activations differ
    by 1 count).
    """
    try:
        return _build_tmt5()
    except Exception as exc:  # noqa: BLE001 - wrap into an error with guidance
        raise RuntimeError(
            "failed to build the fastwam text-encoder-fused CUDA extension "
            f"('{_TMT5_EXT_NAME}'); needs nvcc + a CUDA toolkit matching torch "
            f"(torch {_import_torch_version()}). "
            f"Original error: {exc}"
        ) from exc


def _import_torch_version() -> str:
    import torch

    return f"{torch.__version__}+cu{torch.version.cuda}"


# ext key -> source-file list, shared by the loader and sweeps, so the two can't compile
# different kernels.
SOURCES = {"action": _ACTION_SOURCES, "video": _VIDEO_SOURCES, "tmt5": _TMT5_SOURCES}
