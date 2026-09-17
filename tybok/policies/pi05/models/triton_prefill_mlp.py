"""Fused W8A8 (fp8) GEMM for the pi0.5 prefill MLP.

The prefill MLP (GemmaMLP: gate/up/down, width 2048 -> mlp 16384) uses the
efficient W8A8 recipe:

- **offline** per-channel weight quantization (``quantize_weight``) with the
  transposed fp8 weight pre-computed once (no per-call transpose);
- **per-token** activation quantization (``quantize_activation``);
- a single Triton GEMM whose epilogue applies both scales (``acc * a_scale *
  w_scale``) and writes bf16 directly, so there is no separate dequantize pass.

The wired entry point is ``make_fp8_mlp_forward`` (separate
``quantize_activation`` + ``_w8a8_gemm``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_E4M3 = torch.float8_e4m3fn
_EPILOGUE_SCALE = 448.0  # fp8 e4m3 max


@triton.jit
def _quant_act(x_ptr, x_q_ptr, scale_out_ptr, M, K,
               BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Per-token fp8 quantization, bit-exact to ``quantize_activation``.

    Matches PyTorch's ``x.abs().max(dim=-1)/448`` (which yields **bf16** scale) +
    bf16-rounded ``x / scale`` + ``.to(float8_e4m3fn)`` exactly: the scale is
    computed in fp32 then rounded to bf16, and the division is done in fp32 using
    the *bf16-rounded* scale then rounded to bf16 before the fp8 cast.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    m = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        m = tl.maximum(m, tl.max(tl.abs(x), axis=1))

    scale = (m / 448.0).to(tl.bfloat16)  # bf16, like torch
    tl.store(scale_out_ptr + rows, scale, mask=row_mask)

    s_f32 = scale.to(tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        x = tl.load(x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        y = (x.to(tl.float32) / s_f32[:, None]).to(tl.bfloat16)
        tl.store(x_q_ptr + rows[:, None] * K + ks[None, :], y.to(tl.float8e4nv),
                 mask=row_mask[:, None] & k_mask[None, :])


@triton.jit
def _w8a8_gemm(
    x_ptr, w_ptr, w_scale_ptr, a_scale_ptr, out_ptr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """x [M,K] fp8 @ w [K,N] fp8 -> out [M,N] bf16, dequantized by a_scale * w_scale."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    m_mask = rm < M
    n_mask = rn < N

    a_scale = tl.load(a_scale_ptr + rm, mask=m_mask, other=0.0)  # [BM] fp32
    w_scale = tl.load(w_scale_ptr + rn, mask=n_mask, other=0.0)  # [BN] fp32

    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        k_mask = rk < K
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + rk[:, None] * N + rn[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(x, w)

    out = acc * a_scale[:, None] * w_scale[None, :]
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel quantize weight [N,K] bf16 -> (w_q_T [K,N] fp8, w_scale [N] fp32)."""
    s = w.abs().max(dim=1, keepdim=True).values / _EPILOGUE_SCALE  # [N,1] fp32
    w_q = (w / s).to(_FP8_E4M3)  # [N,K] fp8
    w_q_T = w_q.t().contiguous()  # [K,N] fp8 (transposed layout for the GEMM)
    w_scale = s.squeeze(1).to(torch.float32)  # [N]
    return w_q_T, w_scale


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token quantize activation x [M,K] bf16 -> (x_q [M,K] fp8, a_scale [M] fp32).

    Fused Triton kernel (abs-max reduction + scale + bf16 divide + fp8 cast in one
    pass), bit-exact to the torch reference.
    """
    M, K = x.shape
    x = x.contiguous()
    x_q = torch.empty(M, K, dtype=_FP8_E4M3, device=x.device)
    scale_bf16 = torch.empty(M, dtype=torch.bfloat16, device=x.device)
    _quant_act[(triton.cdiv(M, 32),)](x, x_q, scale_bf16, M, K, BLOCK_M=32, BLOCK_K=256)
    return x_q, scale_bf16.to(torch.float32)


def w8a8_gemm(
    x_q: torch.Tensor, a_scale: torch.Tensor,
    w_q_T: torch.Tensor, w_scale: torch.Tensor,
) -> torch.Tensor:
    """Fused GEMM: pre-quantized x_q @ w_q_T, epilogue applies a_scale * w_scale -> bf16."""
    M, K = x_q.shape
    N = w_q_T.shape[1]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x_q.device)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    _w8a8_gemm[grid](
        x_q, w_q_T, w_scale, a_scale, out, M, N, K,
        BM=128, BN=128, BK=64, num_warps=8, num_stages=3,
    )
    return out


def quantize_mlp(mlp) -> dict:
    """Offline-quantize one GemmaMLP: returns the transposed fp8 weights + scales."""
    gate_w_T, gate_s = quantize_weight(mlp.gate_proj.weight.data)
    up_w_T, up_s = quantize_weight(mlp.up_proj.weight.data)
    down_w_T, down_s = quantize_weight(mlp.down_proj.weight.data)
    return {
        "gate_w_T": gate_w_T, "gate_s": gate_s,
        "up_w_T": up_w_T, "up_s": up_s,
        "down_w_T": down_w_T, "down_s": down_s,
    }


def make_fp8_mlp_forward(mlp):
    """Return an fp8 forward for ``mlp`` (gate/up share the quantized activation)."""
    q = quantize_mlp(mlp)
    from .layers import gelu_pytorch_tanh

    def forward(x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, W] bf16 (B=1 during prefill)
        B, L, W = x.shape
        x2d = x.reshape(-1, W)
        x_q, a_scale = quantize_activation(x2d)

        gate = w8a8_gemm(x_q, a_scale, q["gate_w_T"], q["gate_s"])  # [M, mlp_dim]
        up = w8a8_gemm(x_q, a_scale, q["up_w_T"], q["up_s"])
        act = gelu_pytorch_tanh(gate) * up  # element-wise, bf16

        act2d = act.reshape(-1, act.shape[-1])
        act_q, act_scale = quantize_activation(act2d)
        down = w8a8_gemm(act_q, act_scale, q["down_w_T"], q["down_s"])  # [M, W]

        return down.reshape(B, L, W)

    return forward
