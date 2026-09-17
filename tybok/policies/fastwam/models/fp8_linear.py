"""Base module for fp8-weight Linear (for fastWAM deployment) -- **self-contained**.

All fp8 inference primitives (w8a8 GEMM / per-token activation quantization / fp8 weight
quantization recipe) are defined in this file; it depends only on torch / triton and
**imports no other project module**; all other components depend only on this file.

Base contract (replace the bf16 ``nn.Linear`` GEMM in place with w8a8, matching bf16 torch
inference semantics):

- weights: per-row fp8 e4m3fn + fp32 row scale (``weight`` / ``weight_scale``; bf16 files
  are quantized via ``quantize_to_fp8``, fp8 checkpoints are direct-loaded via ``empty_like`` shells);
- input: bf16 activations -> per-token fp8 quantization (``quantize_act``, amax/448 + RNE, bit-level consistent with host);
- compute: fp8 x fp8 -> fp32 accumulate -> ``acc*sa*sw + bias`` (fp32) -> **bf16 output**;
- the deviation comes only from quantization itself (e4m3 ~3.7%/matrix rel), i.e. the drift tier.

**fp32 high-precision tier** (upstream of attention outputs such as o/cross-o): input allows
fp32 and **retains fp32 precision** for direct quantization -- ``quantize_act`` does per-token
amax/448 + RNE on fp32 values, **a single rounding** straight to the fp8 grid (no bf16 cast
first, no hard quantization against the 448 full scale, avoiding double quantization/double
rounding); points that compute in fp32 or output fp32 SDPA (DiT's o/cross-o, video time-emb
fp32 autocast region) use this tier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

FP8_MAX = 448.0


# Launch-parameter table: one argument group per line, in ABI order (see fp8_gemm.cu). The
# signature carries ``# fmt: skip`` so the formatter leaves the table alone; the body is formatted.
@triton.jit
def _fp8_gemm_kernel(
    a_ptr, w_ptr, sa_ptr, sw_ptr, bias_ptr, residual_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    HAS_RESIDUAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    # weights [N,K] loaded as [BK,BN] (w_tile[k,n]=w[n,k]), avoiding tl.trans's shared-memory transpose
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float8e4nv)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float8e4nv)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    sa = tl.load(sa_ptr + rm, mask=mask_m, other=1.0)
    sw = tl.load(sw_ptr + rn, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + rn, mask=mask_n, other=0.0)
    acc = acc * sa[:, None] * sw[None, :] + bias[None, :]
    if HAS_RESIDUAL:
        res = tl.load(
            residual_ptr + rm[:, None] * N + rn[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc + res  # fp32 residual (higher precision than bf16 x+out)
    out = acc.to(OUT_DTYPE)

    out_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


_DEFAULT_GEMM_CFG = (64, 64, 128, 4)  # BM, BN, BK, num_warps


def fp8_gemm(
    a8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    cfg: tuple = _DEFAULT_GEMM_CFG,
) -> torch.Tensor:
    """a8 [M,K] fp8, w8 [N,K] fp8, sa [M] fp32, sw [N] fp32, bias [N], out [M,N].

    out_dtype (bf16/fp16) determines the epilogue's dequantized output precision;
    residual [M,N] is added to the dequantized result after fp32 accumulation (higher
    precision than bf16 ``x + out``).

    ``cfg`` = (BM, BN, BK, num_warps) or (BM, BN, BK, num_warps, num_stages).
    """
    M, K = a8.shape
    N = w8.shape[0]
    BM, BN, BK, num_warps = cfg[:4]
    num_stages = cfg[4] if len(cfg) > 4 else 3
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    out_dt = tl.float16 if out_dtype == torch.float16 else tl.bfloat16
    _fp8_gemm_kernel[grid](
        a8,
        w8,
        sa,
        sw,
        bias,
        residual if residual is not None else out,
        out,
        M,
        N,
        K,
        a8.stride(0),
        a8.stride(1),
        w8.stride(0),
        w8.stride(1),
        HAS_RESIDUAL=(residual is not None),
        OUT_DTYPE=out_dt,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def _fp32_to_fp8e4m3_rne(y):
    """fp32 -> fp8 e4m3 RNE cast (bit-exact replica of torch float8_e4m3fn). Returns int32 bit pattern.

    The rounding-bias trick ``(ab + 0x7FFFF + lsb) >> 20`` completes the 23->3 bit RNE
    mantissa rounding in one step (the carry propagates naturally into the exponent).

    .. note::
       ``tl.float8e4nv`` (native cvt) **cannot** be substituted directly: PTX does
       ``cvt.rz.f16.f32`` (truncate to fp16) + ``cvt.rn.satfinite.e4m3x2.f16x2``
       (fp16->fp8 RNE), which is double rounding with the first step truncating toward
       zero, giving a ~0.6% biased error on normal values; matching torch's single RNE
       bit-exactly requires bit manipulation.
    """
    b = y.to(tl.int32, bitcast=True)
    sign = (b >> 31) & 1
    ay = tl.abs(y)
    ab = b & 0x7FFFFFFF

    # RNE round to 11 bits (8 exp + 3 mantissa): add 0x7FFFF + the retained LSB, carry propagates into the exponent
    top = (ab + 0x7FFFF + ((ab >> 20) & 1)) >> 20
    e8 = (top >> 3) - 120
    m8 = top & 7

    # subnormal: e8 <= 0 -> value = rint(ay * 512) * 2^-9 (magic-number trick avoids libdevice)
    # (zero also lands here: m_sub = 0, sign bit preserved -> +/-0 correct)
    m_sub = ((ay * 512.0) + 8388608.0) - 8388608.0
    m_sub = m_sub.to(tl.int32)
    sub_carry = m_sub >= 8
    is_sub = e8 <= 0
    e8 = tl.where(is_sub, tl.where(sub_carry, 1, 0), e8)
    m8 = tl.where(is_sub, tl.where(sub_carry, 0, m_sub), m8)

    # overflow -> NaN (e8=15,m8=7); e8==15,m8==7 is already the NaN produced naturally by RNE carry
    overflow = e8 > 15
    e8 = tl.where(overflow, 15, e8)
    m8 = tl.where(overflow, 7, m8)

    return (sign << 7) | (e8 << 3) | m8


@triton.jit
def _act_amax_kernel(x_ptr, raw_max_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """per-token abs-max, K-dimension parallel: grid=(M/BM, K/BK); each CTA reduces its own
    K-slice tile max then atomic_max into raw_max[M] (fp32 atomic max is order-independent
    -> exact max)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = ks < K
    x = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
    m = tl.max(tl.abs(x.to(tl.float32)), axis=1)
    tl.atomic_max(raw_max_ptr + rows, m, mask=row_mask)


@triton.jit
def _act_cast_kernel(x_ptr, raw_max_ptr, x_q_ptr, scale_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """scale = max(raw/448, 1e-12) -> RNE fp8 cast (grid=(M/BM, K/BK), K parallel)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    k_mask = ks < K
    raw = tl.load(raw_max_ptr + rows, mask=row_mask, other=0.0)
    scale = tl.maximum(raw / 448.0, 1e-12)
    if pid_k == 0:
        tl.store(scale_ptr + rows, scale, mask=row_mask)
    x = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
    y = x.to(tl.float32) / scale[:, None]
    bits = _fp32_to_fp8e4m3_rne(y)
    tl.store(
        x_q_ptr + rows[:, None] * K + ks[None, :],
        bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True),
        mask=row_mask[:, None] & k_mask[None, :],
    )


_QUANT_BLOCK_M = 64
_QUANT_BLOCK_K = 512
_QUANT_WARPS = 8


def quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Activation per-token fp8 quantization (Triton, drift tier): x [M,K] -> (a8 fp8, sa fp32).

    K-parallel in two passes: amax-split (atomic reduction) -> cast-split. Bit-level
    consistent with host ``sa=amax/448; a8=RNE(x/sa)`` (max is order-independent, division/RNE
    are replicated pointwise).

    x is bf16 (norm/modulate/ffn upstream); fp32 is also allowed and **retains fp32 direct
    quantization** (SDPA fp32 output of o/cross-o, fp32 compute islands: a single fp32->fp8
    RNE, no pre-rounding to bf16).
    """
    M, K = x.shape
    x = x.contiguous()
    a8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    sa = torch.empty(M, dtype=torch.float32, device=x.device)
    raw = torch.zeros(M, dtype=torch.float32, device=x.device)
    BM, BK, warps = _QUANT_BLOCK_M, _QUANT_BLOCK_K, _QUANT_WARPS
    grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))
    _act_amax_kernel[grid](x, raw, M, K, BLOCK_M=BM, BLOCK_K=BK, num_warps=warps)
    _act_cast_kernel[grid](x, raw, a8, sa, M, K, BLOCK_M=BM, BLOCK_K=BK, num_warps=warps)
    return a8, sa


# per-shape fp8 GEMM kernel configs: key=(M,K,N), default (64,64,128,4); cfg=(BM,BN,BK,num_warps[,num_stages]).
# wide matrices with N>=6144 are better under BM32/BN128; video FFN-down uses num_stages=4 (deeper K pipeline is more stable).
_GEMM_CFG_BY_SHAPE: dict[tuple[int, int, int], tuple] = {
    # video expert (prefill, M=120 tokens / 129 context)
    (120, 3072, 14336): (128, 64, 128, 4, 3),
    (120, 14336, 3072): (64, 64, 128, 4, 4),
    (120, 3072, 3072): (64, 64, 128, 4),
    (120, 3072, 9216): (32, 128, 128, 4),
    (129, 3072, 3072): (32, 128, 128, 4),
    (129, 3072, 6144): (32, 128, 128, 4),
    (129, 4096, 3072): (32, 128, 128, 4),
    (120, 3072, 6144): (64, 64, 128, 4),
    # action expert (denoise, M=32 suffix / 129 context)
    (32, 1024, 3072): (64, 64, 64, 4),
    (32, 1024, 9216): (32, 64, 128, 4),
    (32, 3072, 1024): (32, 64, 128, 4),
    (32, 1024, 4096): (64, 64, 256, 4),
    (32, 4096, 1024): (32, 64, 128, 4),
    (129, 1024, 3072): (32, 128, 128, 4),
    (129, 1024, 6144): (32, 128, 128, 4),
    (129, 4096, 1024): (64, 64, 128, 4),
}


def pick_gemm_cfg(m: int, k: int, n: int) -> tuple:
    """Select fp8 GEMM kernel params by (M, K, N) (static table + default fallback).

    Returns (BM, BN, BK, num_warps[, num_stages]), directly passable to ``fp8_gemm``.
    """
    return _GEMM_CFG_BY_SHAPE.get((m, k, n), _DEFAULT_GEMM_CFG)


class FP8Linear(nn.Module):
    """Linear with fp8 weights: activation per-token quantization + Triton fp8 GEMM + bias, bf16 output.

    Input bf16 (norm/modulate/ffn upstream) or fp32 (fp32-precision upstream such as SDPA
    fp32 output) -> per-token fp8 direct quantization -> fp32 accumulate -> bf16 output.
    An fp8/integer input is treated as a wiring error (the upstream already quantized once,
    i.e. double quantization) and raises directly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.weight_scale = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FP8Linear:
        w = linear.weight.data.float()
        scale = (w.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
        w8 = (w / scale[:, None]).to(torch.float8_e4m3fn).contiguous()
        m = cls(linear.in_features, linear.out_features, linear.bias is not None)
        m.weight = nn.Parameter(w8)
        m.weight_scale = nn.Parameter(scale)
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.data.to(torch.bfloat16))
        else:
            m.register_parameter("bias", None)
        return m

    @classmethod
    def empty_like(cls, linear: nn.Linear) -> FP8Linear:
        """An empty FP8Linear that copies only the structure (weights/scale uninitialized).

        For "fp8 file direct-load": before ``load_model``, replace the ``nn.Linear`` set
        matching the fp8 file with shells, then the loader copies the fp8 weights and per-row
        scale directly from ``*.fp8.safetensors`` (skipping the dequant->requant double
        rounding, and avoiding materializing 12 GB of bf16 first).
        """
        return cls(linear.in_features, linear.out_features, linear.bias is not None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                f"FP8Linear input must be bf16 (base contract) or fp32 (high-precision direct-quant "
                f"tier: o/cross-o's SDPA output / fp32 compute islands), got {x.dtype}; an fp8 input "
                f"means the upstream already quantized once (double-quantization wiring error)"
            )
        # per-token fp8 quantization + w8a8 GEMM; quantization semantics are bit-level
        # consistent with host amax/448 + RNE cast; fp32 input takes **direct quantization**
        # (a single fp32->fp8 RNE rounding, no pre-rounding to bf16).
        lead = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features)
        M = x2d.shape[0]
        a8, sa = quantize_act(x2d)
        out = torch.empty(M, self.out_features, dtype=torch.bfloat16, device=x.device)
        bias = (
            self.bias
            if self.bias is not None
            else torch.zeros(self.out_features, dtype=torch.bfloat16, device=x.device)
        )
        fp8_gemm(
            a8, self.weight, sa, self.weight_scale, bias, out, cfg=pick_gemm_cfg(M, self.in_features, self.out_features)
        )
        return out.reshape(*lead, self.out_features)


def quantize_to_fp8(module: nn.Module, min_dim: int = 256) -> int:
    """Recursively replace nn.Linear with max(in, out) >= min_dim by FP8Linear; returns the replacement count.

    Large matrices only: the fp8 benefit comes from halving weight bytes (memory-bound GEMM),
    while medium matrices may actually be slower than cuBLAS; the video expert uses
    min_dim=3072 to quantize only FFN(14336) and time_projection(18432).
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and max(child.in_features, child.out_features) >= min_dim:
            setattr(module, name, FP8Linear.from_linear(child))
            count += 1
        else:
            count += quantize_to_fp8(child, min_dim)
    return count


def fp8ify_structural(module: nn.Module, min_dim: int = 256) -> int:
    """Recursively replace nn.Linear with **both dimensions** >= min_dim by empty FP8Linear shells (no quantization); returns the count.

    Consistent with the quantization rule of ``model.fp8.safetensors``: matrices where only
    one of in/out is large (e.g. video head Linear: 192x3072) stay bf16 in the file, so the
    bf16 Linear must be kept for the loader to load as-is, and must not be replaced by an
    fp8 shell.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= min_dim and child.out_features >= min_dim:
            setattr(module, name, FP8Linear.empty_like(child))
            count += 1
        else:
            count += fp8ify_structural(child, min_dim)
    return count
