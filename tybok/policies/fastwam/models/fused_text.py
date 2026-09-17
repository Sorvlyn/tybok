"""Single-kernel fusion of the UMT5 text encoder's **two sublayers** (`--cu-fused-text-encoder`).

Attention sublayer (`FusedAttnRunner`): replaces `UMT5LayerSelfAttention.fused_forward`'s
Triton chain (norm+quantize -> qkv fp8 GEMM -> eager attention -> o-proj fp8 GEMM+residual)
with one cooperative launch, 5 phase / 4 grid.sync.

FFN sublayer (`FusedTextRunner`): replaces the Triton three-op chain in
`UMT5LayerFF.fused_forward`

    wi  = fused_norm_wi_gemm(x)      # RMSNorm + packed (wi_0|wi_1) fp8 GEMM
    act = gelu_new_gated(wi)         # gelu_new(wi_0) * wi_1
    out = wo_residual(act, x)        # wo fp8 GEMM + residual

with one cooperative CUDA kernel `tmt5.ffn` (4 phases / 3 grid.sync):
RMSNorm+quantize -> packed-wi GEMM -> gelu_new gating+quantize -> wo GEMM+residual.

Numeric tier: **bit-aligned with the existing Triton chain**. `tmt5.ffn`/P2 (packed-wi GEMM),
/P3 (gating+quantize) and /P4 (wo+residual) are **bit-identical** to the production chain;
/P1 (RMSNorm)'s reduction order differs from Triton's `tl.sum` tree, giving end-to-end
rel-RMS ~1e-3.
"""

from __future__ import annotations

import torch

from ..kernels import load_tmt5_fused_ext, phases

# matches the CUDA kernel compile-time constants (TMT5_M / TMT5_H / TMT5_F in tmt5_ffn.cu)
_M = 128
_H = 4096
_F = 10240

_FP8 = torch.float8_e4m3fn


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"[text-fused] {msg}")


# ------------------------------------------------------------------ #
# optional: pin inter-phase intermediates (scratch) in L2 (persisting access policy window)
#
# Each layer's scratch is isolated between phases by grid.sync and can only cross CTAs via
# global, so after write-back it may be evicted to DRAM by the weight stream. Putting the
# whole scratch region in one contiguous arena and attaching a persisting
# accessPolicyWindow to the current stream lets it be reused across layers in L2.
#
# Affects cache policy only, changes no numeric result (bit-identical). Off by default.
# ------------------------------------------------------------------ #
class _ScratchArena:
    """Contiguous scratch allocator (the L2 window requires a single contiguous address range)."""

    _ALIGN = 1024

    def __init__(self, device, capacity: int):
        self.device = torch.device(device)
        self._buf = torch.empty(int(capacity), dtype=torch.uint8, device=self.device)
        self._off = 0

    def tensor(self, dtype: torch.dtype, shape, zero: bool = False) -> torch.Tensor:
        numel = 1
        for s in shape:
            numel *= int(s)
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()
        self._off = (self._off + self._ALIGN - 1) // self._ALIGN * self._ALIGN
        start = self._off
        self._off += nbytes
        _require(
            self._off <= self._buf.numel(), f"scratch arena too small (need {self._off} > {self._buf.numel()} bytes)"
        )
        t = self._buf[start : start + nbytes].view(dtype).view(*shape)
        if zero:
            t.zero_()
        return t

    @property
    def used(self) -> int:
        return self._off

    def data_ptr(self) -> int:
        return self._buf.data_ptr()


def enable_l2_persist(arena: _ScratchArena, limit_bytes: int | None = None):
    """Mark the address range covered by the arena as L2 persisting (current stream). Returns raw values for logging."""
    from cuda.bindings import runtime as rt

    max_persist = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize, 0)[1]
    max_window = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize, 0)[1]
    n = min(int(arena.used), int(max_window))
    limit = min(n if limit_bytes is None else int(limit_bytes), int(max_persist))
    rt.cudaDeviceSetLimit(rt.cudaLimit.cudaLimitPersistingL2CacheSize, limit)
    v = rt.cudaStreamAttrValue()
    v.accessPolicyWindow.base_ptr = arena.data_ptr()
    v.accessPolicyWindow.num_bytes = n
    v.accessPolicyWindow.hitRatio = 1.0
    v.accessPolicyWindow.hitProp = rt.cudaAccessProperty.cudaAccessPropertyPersisting
    v.accessPolicyWindow.missProp = rt.cudaAccessProperty.cudaAccessPropertyStreaming
    stream = torch.cuda.current_stream().cuda_stream
    rt.cudaStreamSetAttribute(stream, rt.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow, v)
    return dict(window_bytes=n, limit_bytes=limit, max_persist=int(max_persist))


def disable_l2_persist(arena: _ScratchArena) -> None:
    """Reset the window to normal (to avoid affecting subsequent other kernels)."""
    from cuda.bindings import runtime as rt

    v = rt.cudaStreamAttrValue()
    v.accessPolicyWindow.base_ptr = arena.data_ptr()
    v.accessPolicyWindow.num_bytes = min(
        int(arena.used), int(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize, 0)[1])
    )
    v.accessPolicyWindow.hitRatio = 0.0
    v.accessPolicyWindow.hitProp = rt.cudaAccessProperty.cudaAccessPropertyNormal
    v.accessPolicyWindow.missProp = rt.cudaAccessProperty.cudaAccessPropertyNormal
    rt.cudaStreamSetAttribute(
        torch.cuda.current_stream().cuda_stream, rt.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow, v
    )
    rt.cudaCtxResetPersistingL2Cache()


def _alloc(arena, device, dtype, shape, zero: bool = False) -> torch.Tensor:
    if arena is not None:
        return arena.tensor(dtype, shape, zero=zero)
    if zero:
        return torch.zeros(*shape, dtype=dtype, device=device)
    return torch.empty(*shape, dtype=dtype, device=device)


class FusedTextRunner:
    """Point each UMT5 layer's FFN sublayer at the fused kernel.

    Usage (construct once in the engine):

        runner = FusedTextRunner(encoder.encoder, device)
        runner.install(encoder.encoder)

    Afterward each `UMT5LayerFF.fused_forward` calls the fused kernel; weights are
    referenced per layer, not copied.
    """

    def __init__(self, stack, device: torch.device, arena=None, split: bool = False):
        blocks = list(stack.block)
        _require(len(blocks) > 0, "text encoder has no blocks")
        self.device = device
        # split=True: non-cooperative form -- the FFN kernel is split into 4 plain launches
        # (not requiring whole-GPU co-residency); split=False: 1 cooperative launch.
        # Numerically bit-identical.
        self._split = bool(split)
        self.ext = load_tmt5_fused_ext()
        self.w = [self._resolve(blk) for blk in blocks]
        # output ping-pong: `x_in` is also the `tmt5.ffn`/P4 residual term, so writing into the same
        # buffer would clobber the read. **A single global toggle** (not one per layer):
        # layer i writes _out[ob] and returns; layer i+1 reads it as x_in and writes
        # _out[1-ob]. If each layer started from 0, layer i+1's x_in and its out would be
        # the same buffer.
        self._tog = 0
        self.arena = arena

        self._a8 = _alloc(arena, device, torch.uint8, (_M, _H))
        self._sa0 = _alloc(arena, device, torch.float32, (_M,))
        self._gbuf = _alloc(arena, device, torch.bfloat16, (_M, 2 * _F))
        self._a8g = _alloc(arena, device, torch.uint8, (_M, _F))
        self._sa1 = _alloc(arena, device, torch.float32, (_M,))
        # bias all zeros: production `fused_norm_wi_gemm` / `wo_residual` also pass
        # `torch.zeros`, and the epilogue's `acc*sa*sw + 0.0f` is not equivalent to passing
        # no bias at signed zero (-0+0=+0), so this step must really run; do not pass
        # nullptr to the kernel.
        self._bias_wi = _alloc(arena, device, torch.bfloat16, (2 * _F,), zero=True)
        self._bias_wo = _alloc(arena, device, torch.bfloat16, (_H,), zero=True)
        self._out = [_alloc(arena, device, torch.bfloat16, (_M, _H)) for _ in range(2)]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve(blk) -> dict:
        """Fetch one layer's fp8 FFN weights / scale (all references, no copies).

        Note `pack_fused(ffn=True)` has already packed wi_0/wi_1 into `wi_weight` and freed
        the original weights (`UMT5DenseGatedActDense.pack_wi`), so this reads the packed
        copy -- the same tensor the Triton path consumes.
        """
        ffn = blk.layer[1]
        f = ffn.DenseReluDense
        _require(
            getattr(f, "wi_weight", None) is not None,
            "DenseReluDense has no wi_weight (--cu-fused-text-encoder requires pack_fused(ffn=True))",
        )
        _require(f.wi_weight.dtype == _FP8 and f.wi_weight.is_contiguous(), "wi_weight requires contiguous fp8 e4m3fn")
        _require(f.wo.weight.dtype == _FP8 and f.wo.weight.is_contiguous(), "wo.weight requires contiguous fp8 e4m3fn")
        _require(
            f.wi_scale.dtype == torch.float32
            and f.wi_scale.is_contiguous()
            and f.wo.weight_scale.dtype == torch.float32
            and f.wo.weight_scale.is_contiguous(),
            "scale requires contiguous fp32",
        )
        nw = ffn.layer_norm.weight
        _require(
            nw is not None and nw.dtype == torch.bfloat16 and nw.is_contiguous(),
            "layer_norm.weight must be contiguous bf16 (kernel reads as bf16)",
        )
        _require(
            f.wi_weight.shape == (2 * _F, _H) and f.wo.weight.shape == (_H, _F),
            f"FFN geometry mismatch (kernel compile-time fixed {_H}->{2 * _F}->{_H})",
        )
        return dict(w0=f.wi_weight, sw0=f.wi_scale,
                    w1=f.wo.weight.data, sw1=f.wo.weight_scale.data,
                    nw=nw, eps=float(ffn.layer_norm.variance_epsilon))  # fmt: skip

    # ------------------------------------------------------------------ #
    def _ffn(self, i: int):
        w = self.w[i]

        def ffn_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            lead = hidden_states.shape[:-1]
            x2 = hidden_states.reshape(-1, _H)
            _require(
                x2.shape[0] == _M,
                f"token count {x2.shape[0]} != kernel compile-time {_M} (UMT5 pads to fixed length 128)",
            )
            _require(x2.is_contiguous(), "hidden_states is not contiguous")
            ob = self._tog
            self._tog ^= 1
            stream = torch.cuda.current_stream().cuda_stream
            # phases go through the registry (default impl = the original ext call, verbatim; see kernels/phases.py)
            tensors = dict(
                x_in=x2, nw=w["nw"], w0=w["w0"], sw0=w["sw0"], b0=self._bias_wi,
                w1=w["w1"], sw1=w["sw1"], b1=self._bias_wo,
                out=self._out[ob], a8buf=self._a8, sa0buf=self._sa0,
                gbuf=self._gbuf, a8gbuf=self._a8g, sa1buf=self._sa1,
            )  # fmt: skip
            scalars = dict(stop_phase=0, norm_eps=w["eps"], phase_ts=0)
            for ph in range(1, 5) if self._split else (-1,):
                phases.dispatch("tmt5.ffn", ph, tensors=tensors, scalars=scalars, stream=stream)
            return self._out[ob].reshape(*lead, _H)

        return ffn_forward

    # ------------------------------------------------------------------ #
    def install(self, stack) -> None:
        """Attach the hook to each layer's FFN sublayer (`UMT5LayerFF.fused_forward` checks it)."""
        blocks = list(stack.block)
        _require(len(blocks) == len(self.w), "block count differs from construction")
        for i, blk in enumerate(blocks):
            blk.layer[1]._fused_ffn = self._ffn(i)


class FusedAttnRunner:
    """Point each UMT5 layer's **self-attention sublayer** at the fused kernel (the other half of `--cu-fused-text-encoder`).

    Covers `UMT5LayerSelfAttention.fused_forward`'s production Triton chain
    (norm+qkv GEMM -> eager attention -> o-proj GEMM+residual, 5 phase / 4 grid.sync):

        `tmt5.attn`/P1 RMSNorm+per-token fp8 quantize -> /P2 packed-qkv fp8 GEMM -> /P3 bf16
        attention (mma QK^T -> +pos_bias+causal -> fp32 softmax -> mma PV) -> /P4 quantize -> /P5 o GEMM+residual

    Usage (construct once in the engine):

        runner = FusedAttnRunner(encoder.encoder, device)
        runner.install(encoder.encoder)

    Afterward each `UMT5LayerSelfAttention.fused_forward` calls the fused kernel; weights
    are referenced per layer, not copied.
    """

    def __init__(self, stack, device: torch.device, arena=None, split: bool = False):
        blocks = list(stack.block)
        _require(len(blocks) > 0, "text encoder has no blocks")
        self.device = device
        # split=True: non-cooperative form -- the attention kernel is split into 5 plain launches (not requiring whole-GPU co-residency).
        self._split = bool(split)
        self.ext = load_tmt5_fused_ext()
        self.w = [self._resolve(blk) for blk in blocks]
        # output ping-pong: `out`'s residual term is x_in, so writing into the same buffer would clobber the read (same as FusedTextRunner).
        self._tog = 0
        self.arena = arena

        self._a8 = _alloc(arena, device, torch.uint8, (_M, _H))
        self._sa0 = _alloc(arena, device, torch.float32, (_M,))
        self._qkv = _alloc(arena, device, torch.bfloat16, (_M, 3 * _H))
        self._attnb = _alloc(arena, device, torch.bfloat16, (_M, _H))
        self._a8o = _alloc(arena, device, torch.uint8, (_M, _H))
        self._sa1 = _alloc(arena, device, torch.float32, (_M,))
        # bias all zeros: same recipe as the `torch.zeros` passed by production
        # `fused_norm_qkv_gemm` / `o_proj_residual` (the epilogue's `acc*sa*sw + 0.0f` is
        # not equivalent to passing no bias at signed zero).
        self._bqkv = _alloc(arena, device, torch.bfloat16, (3 * _H,), zero=True)
        self._bo = _alloc(arena, device, torch.bfloat16, (_H,), zero=True)
        self._out = [_alloc(arena, device, torch.bfloat16, (_M, _H)) for _ in range(2)]

        # make each layer's pos_bias a **contiguous** copy (+2MB/layer): `_pos_bias` is the
        # permute result of `_compute_bias` (column stride 64 elements, non-contiguous),
        # while the kernel copies the whole row-major [S,S] into smem in 16B chunks.
        for i, blk in enumerate(blocks):
            pb = blk.layer[0].SelfAttention._pos_bias
            _require(
                pb is not None,
                "SelfAttention has no _pos_bias (--cu-fused-text-encoder requires pack_fused(attn=True))",
            )
            self.w[i]["pos"] = pb.contiguous()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve(blk) -> dict:
        """Fetch one layer's fp8 qkv/o weights / scale / norm weight (all references, no copies).

        ``blk`` is a `UMT5Block` (same convention as `FusedTextRunner._resolve`); the
        attention sublayer is at `blk.layer[0]`.
        """
        layer = blk.layer[0]
        sa = layer.SelfAttention
        _require(getattr(sa, "qkv_weight", None) is not None, "SelfAttention has no qkv_weight (pack_qkv not run)")
        _require(
            sa.qkv_weight.dtype == _FP8 and sa.qkv_weight.is_contiguous(), "qkv_weight requires contiguous fp8 e4m3fn"
        )
        _require(sa.o.weight.dtype == _FP8 and sa.o.weight.is_contiguous(), "o.weight requires contiguous fp8 e4m3fn")
        _require(
            sa.qkv_scale.dtype == torch.float32
            and sa.qkv_scale.is_contiguous()
            and sa.o.weight_scale.dtype == torch.float32
            and sa.o.weight_scale.is_contiguous(),
            "scale requires contiguous fp32",
        )
        nw = layer.layer_norm.weight
        _require(
            nw is not None and nw.dtype == torch.bfloat16 and nw.is_contiguous(),
            "layer_norm.weight must be contiguous bf16 (kernel reads as bf16)",
        )
        _require(
            sa.qkv_weight.shape == (3 * _H, _H) and sa.o.weight.shape == (_H, _H),
            f"attention geometry mismatch (kernel compile-time fixed {_H}->{3 * _H}->{_H})",
        )
        return dict(wq=sa.qkv_weight, swq=sa.qkv_scale,
                    wo=sa.o.weight.data, swo=sa.o.weight_scale.data,
                    nw=nw, eps=float(layer.layer_norm.variance_epsilon))  # fmt: skip

    # ------------------------------------------------------------------ #
    def _attn(self, i: int):
        w = self.w[i]

        def attn_forward(
            hidden_states: torch.Tensor, pos_bias: torch.Tensor, causal_mask: torch.Tensor
        ) -> torch.Tensor:
            lead = hidden_states.shape[:-1]
            x2 = hidden_states.reshape(-1, _H)
            _require(
                x2.shape[0] == _M,
                f"token count {x2.shape[0]} != kernel compile-time {_M} (UMT5 pads to fixed length 128)",
            )
            _require(x2.is_contiguous(), "hidden_states is not contiguous")
            cm = causal_mask.reshape(-1)  # [B,1,1,S] -> [S] bf16
            _require(cm.numel() == _M, f"causal_mask length {cm.numel()} != {_M}")
            _require(cm.dtype == torch.bfloat16, "causal_mask must be bf16")
            ob = self._tog
            self._tog ^= 1
            stream = torch.cuda.current_stream().cuda_stream
            # phases go through the registry (default impl = the original ext call, verbatim)
            tensors = dict(
                x_in=x2, nw=w["nw"], wqkv=w["wq"], sqkv=w["swq"], bqkv=self._bqkv,
                wo=w["wo"], swo=w["swo"], bo=self._bo, posb=w["pos"], amask=cm,
                out=self._out[ob], a8buf=self._a8, sa0buf=self._sa0,
                qkvbuf=self._qkv, attnbuf=self._attnb, a8obuf=self._a8o, sa1buf=self._sa1,
            )  # fmt: skip
            scalars = dict(stop_phase=0, norm_eps=w["eps"], phase_ts=0, dbg=0)
            for ph in range(1, 6) if self._split else (-1,):
                phases.dispatch("tmt5.attn", ph, tensors=tensors, scalars=scalars, stream=stream)
            return self._out[ob].reshape(*lead, _H)

        return attn_forward

    # ------------------------------------------------------------------ #
    def install(self, stack) -> None:
        """Attach the hook to each layer's self-attention sublayer (`UMT5LayerSelfAttention.fused_forward` checks it)."""
        blocks = list(stack.block)
        _require(len(blocks) == len(self.w), "block count differs from construction")
        for i, blk in enumerate(blocks):
            blk.layer[0]._fused_attn = self._attn(i)
