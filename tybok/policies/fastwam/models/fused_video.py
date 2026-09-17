"""Single-kernel fusion of the video expert FFN tail (`--cu-fused-vdit`).

Replace this block-tail segment

    mlp_input = modulate(apply_norm2(x), shift_mlp, scale_mlp)   # WanLayerNorm + AdaLN
    out       = x + gate_mlp * ffn(mlp_input)                    # ffn = up->GELU->down

with a single cooperative CUDA kernel `vdit.ffn` (4 phases / 3 grid.sync):
input quantize -> up GEMM -> GELU -> quantize -> down GEMM -> gate residual.

**Attention here still goes through torch** (the video attn chain is not fused), so
this runner does not take over the whole layer; it only replaces the FFN tail via
a hook on `FastWAMAttentionBlock`:

    blk._fused_ffn_tail(x, shift_mlp, scale_mlp, gate_mlp) -> out

Numeric tier: **quantization drift tier** (rel <= 3e-2, same order as fp8
quantization noise itself). Sources of difference vs the production torch/Triton
chain: GELU uses torch's tanh form -- `nn.GELU(approximate='tanh')`, i.e.
0.5x(1+tanh(c1(x+c2x^3))) with c1=sqrt(2/pi), the same as lerobot's
`policies/fastwam/wan` (the shared helper in `kernels/common.h` used to drop the c1
factor on the cubic term, a systematic 6.3e-3 max deviation; fixed 2026-09-14) --
fp8 uses software single-pass RNE, and norm2 mean/var use the same fp32 statistics but a
different reduction order.
Bit-exact regression is impossible; acceptance is judged by rel.
"""

from __future__ import annotations

import torch

from ..kernels import load_video_fused_ext, phases

# matches the CUDA kernel compile-time constants (VF_M / VF_H / VF_F in vdit_ffn.cu)
_M = 120
_H = 3072
_F = 14336

_FP8 = torch.float8_e4m3fn


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"[video-fused] {msg}")


class FusedVideoRunner:
    """Point each video expert layer's FFN tail at the fused kernel.

    Usage (construct once in the engine):

        runner = FusedVideoRunner(blocks, device)
        runner.install(blocks)

    Afterward each time `block.forward` reaches the FFN tail it calls the fused
    kernel; weights are referenced per layer, not copied.
    """

    def __init__(self, blocks, device: torch.device, split: bool = False):
        n = len(blocks)
        _require(n > 0, "video expert has no blocks")
        self.device = device
        # split=True: non-cooperative fallback -- each layer's FFN tail is split into 4
        # plain kernel launches (vdit.ffn/P1..P4), for when a cooperative launch cannot get
        # whole-GPU co-residency (small GPUs / MPS shared partitions). Intra-phase fusion is unchanged.
        self._split = bool(split)
        self.ext = load_video_fused_ext()
        self.w = [self._resolve(blk) for blk in blocks]
        self._tog = [0] * n  # output buffer ping-pong (avoids in-place read/write with the residual term)

        # scratch (shaped per the kernel interface; fp8 stored as uint8, data_ptr passed directly)
        self._a8 = torch.empty(_M, _H, dtype=torch.uint8, device=device)
        self._sa0 = torch.empty(_M, dtype=torch.float32, device=device)
        self._gbuf = torch.empty(_M, _F, dtype=torch.bfloat16, device=device)
        self._a8g = torch.empty(_M, _F, dtype=torch.uint8, device=device)
        self._sa1 = torch.empty(_M, dtype=torch.float32, device=device)
        self._raw = torch.empty(_M, dtype=torch.uint32, device=device)
        self._out = [torch.empty(_M, _H, dtype=torch.bfloat16, device=device) for _ in range(2)]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve(blk) -> dict[str, torch.Tensor]:
        """Fetch one layer's fp8 FFN weights / scale / bias (all references, no copies)."""
        from .fp8_linear import FP8Linear

        for name, m in (("ffn.0", blk.ffn[0]), ("ffn.2", blk.ffn[2])):
            _require(
                isinstance(m, FP8Linear),
                f"{name} is not FP8Linear (--cu-fused-vdit needs the video expert fp8-resident, "
                "i.e. --video-bf16 cannot be used together with it)",
            )
            _require(m.weight.dtype == _FP8, f"{name}.weight is not fp8 e4m3fn")
            _require(m.weight.is_contiguous(), f"{name}.weight is not contiguous")
            _require(
                m.weight_scale.dtype == torch.float32 and m.weight_scale.is_contiguous(),
                f"{name}.weight_scale requires contiguous fp32",
            )
            _require(
                m.bias is not None and m.bias.dtype == torch.bfloat16 and m.bias.is_contiguous(),
                f"{name} requires a contiguous bf16 bias",
            )
        _require(
            blk.ffn[0].in_features == _H
            and blk.ffn[0].out_features == _F
            and blk.ffn[2].in_features == _F
            and blk.ffn[2].out_features == _H,
            f"FFN geometry mismatch (kernel compile-time fixed {_H}->{_F}->{_H})",
        )
        _require(blk.norm2.weight is None, "norm2 must be elementwise_affine=False (Wan default)")
        return dict(
            w0=blk.ffn[0].weight, sw0=blk.ffn[0].weight_scale, b0=blk.ffn[0].bias,
            w1=blk.ffn[2].weight, sw1=blk.ffn[2].weight_scale, b1=blk.ffn[2].bias,
            eps=float(blk.norm2.eps),
        )  # fmt: skip

    # ------------------------------------------------------------------ #
    def _tail(self, i: int):
        w = self.w[i]

        def tail(
            x: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor
        ) -> torch.Tensor:
            lead = x.shape[:-1]
            x2d = x.reshape(-1, _H)
            _require(
                x2d.shape[0] == _M,
                f"token count {x2d.shape[0]} != kernel compile-time {_M} (video prefill only supports S={_M})",
            )
            _require(x2d.is_contiguous(), "x is not contiguous")

            # shift/scale/gate come from split_modulation. video's t_mod is 4-D
            # (time_projection's unflatten(2) stays at [B,S,6,D]) and takes the has_seq
            # branch -> each is [B,S,1,D].squeeze(2) = **[B,S,H], one modulation set per
            # token**; and it is a **view from chunk(dim=2), with row stride 6H not H**, so
            # reshape/contiguous would materialize a copy.
            # So pass the real row stride to the kernel here: mod_srow = stride(-2); for a
            # broadcast shape (1 row) pass 0.
            def _mod(t):
                _require(t.shape[-1] == _H, "the last dim of shift/scale/gate must be hidden")
                if t.dim() < 2:
                    return t, 1, 0
                return t, t.shape[-2], t.stride(-2)

            sh, nrow_h, srow_h = _mod(shift_mlp)
            sc, nrow_c, srow_c = _mod(scale_mlp)
            gt, nrow_g, srow_g = _mod(gate_mlp)
            _require(
                nrow_h == nrow_c == nrow_g and nrow_h in (1, _M),
                f"the row count of shift/scale/gate must be 1 or {_M} (got {nrow_h})",
            )
            _require(srow_h == srow_c == srow_g, "shift/scale/gate must have the same row stride")
            mod_srow = 0 if nrow_h == 1 else srow_h

            o = self._out[self._tog[i]]
            self._tog[i] ^= 1
            # non-cooperative fallback: split into 4 plain launches (vdit.ffn/P1..P4), ordered
            # between phases by stream order; the cooperative path (split=False) is still
            # 1 cooperative launch. only_phase=-1 = full pipeline.
            tensors = dict(
                x_in=x2d, gate_mlp=gt, shift_mlp=sh, scale_mlp=sc,
                w0=w["w0"], sw0=w["sw0"], b0=w["b0"],
                w1=w["w1"], sw1=w["sw1"], b1=w["b1"],
                out=o, a8buf=self._a8, sa0buf=self._sa0,
                gbuf=self._gbuf, a8gbuf=self._a8g, sa1buf=self._sa1, raw1=self._raw,
            )  # fmt: skip
            scalars = dict(stop_phase=0, use_norm=1, norm_eps=w["eps"], phase_ts=0, mod_srow=mod_srow)
            for ph in range(1, 5) if self._split else (-1,):
                phases.dispatch(
                    "vdit.ffn", ph, tensors=tensors, scalars=scalars, stream=torch.cuda.current_stream().cuda_stream
                )
            return o.reshape(*lead, _H)

        return tail

    # ------------------------------------------------------------------ #
    def install(self, blocks) -> None:
        """Attach the FFN-tail hook to each layer (block forward checks it before the FFN)."""
        _require(len(blocks) == len(self.w), "block count differs from construction")
        for i, blk in enumerate(blocks):
            blk._fused_ffn_tail = self._tail(i)


# ===================================================================== #
# attention-segment fusion (the other half of `--cu-fused-vdit`): self + cross fused kernels
# ===================================================================== #
def _video_rope_tables(freqs_table: torch.Tensor, grid) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side cos/sin tables for 3-D grid RoPE (fp32 [S,64]).

    `_build_freqs` builds complex128 `[1024,64]`; `rope_apply` splits it as
    `[c-2(c//3), c//3, c//3]` into f/h/w groups, expands each along the corresponding
    grid dimension, then cats them back -- so the **column index is unchanged, only the
    row index switches with the group**, and this is just a column-grouped gather.
    Group boundaries are derived from c (do not hardcode 22/21/21; changing head_dim
    would break it).
    """
    f, h, w = (int(v) for v in grid)
    S = f * h * w
    idx = torch.arange(S)
    fi = idx // (h * w)
    hi = (idx // w) % h
    wi = idx % w
    c = freqs_table.shape[-1]
    g1 = c - 2 * (c // 3)
    g2 = c // 3
    tab = torch.empty(S, c, dtype=torch.complex128)
    for lo, hi_, row in ((0, g1, fi), (g1, g1 + g2, hi), (g1 + g2, c, wi)):
        tab[:, lo:hi_] = freqs_table[row][:, lo:hi_]
    return tab.real.float().contiguous(), tab.imag.float().contiguous()


class FusedAttnRunner:
    """Point **the entire attention segment** (self + cross) of each video expert layer
    at two cooperative kernels.

    Replaces the following in `FastWAMAttentionBlock.forward`:

        attn_input = modulate(apply_norm1(x), shift_msa, scale_msa)
        q, k, v    = project_self_attention(attn_input, freqs)      # qkv + qk-norm + 3-D RoPE
        x          = x + gate_msa * o(SDPA(q, k, v))
        x          = x + cross_attn(norm3(x), context)

    Numeric tier: **quantization drift tier** (same tier as the FFN tail). Sources of
    difference vs the reference: flash uses bf16 mma (the reference is fp32 SDPA, but
    q/k/v are already the bf16 output of an fp8 GEMM), RoPE is computed in fp32
    (reference fp64), and qk-norm's mean-square uses a warp-tree reduction (the
    reference uses torch's reduction order). Note the attn output stays **fp32** and is
    quantized directly -- the reference o-proj also quantizes from fp32 with a single
    RNE, so this has one fewer bf16 rounding than the action fused kernel.

    Serves **video prefill only** (M=120, context all-True mask); the action-denoise
    `video_kv`/`context_kv` paths are rejected outright to avoid silently taking the
    wrong branch.
    """

    _M = 120
    _H = 3072
    _H3 = 3072
    _C = 129

    def __init__(self, blocks, device, eps: float = 1e-6, split: bool = False):
        n = len(blocks)
        _require(n > 0, "video expert has no blocks")
        self.device = device
        self.eps = float(eps)
        # split=True: non-cooperative fallback -- each layer's self/cross is split into
        # 6 plain kernel launches, for when a cooperative launch cannot get whole-GPU
        # co-residency (small GPUs / MPS shared partitions). Intra-phase fusion is unchanged.
        self._split = bool(split)
        self.ext = load_video_fused_ext()
        self.w = [self._resolve(b) for b in blocks]

        d = device
        e = lambda *s, dt: torch.empty(*s, dtype=dt, device=d)  # noqa: E731
        self._qkv = e(self._M, 9216, dt=torch.bfloat16)
        self._qc = e(self._M, self._H3, dt=torch.bfloat16)
        # k/v **one per layer**: they are collected by prefill as that layer's KV cache
        # and read repeatedly by the following 10 denoise steps; sharing one buffer would
        # make all 30 layers point at the last layer.
        self._kc = [e(self._M, self._H3, dt=torch.bfloat16) for _ in range(n)]
        self._vc = [e(self._M, self._H3, dt=torch.bfloat16) for _ in range(n)]
        self._attn = e(self._M, self._H3, dt=torch.float32)
        self._a8x = e(self._M, self._H, dt=torch.uint8)
        self._sa0x = e(self._M, dt=torch.float32)
        self._a8a = e(self._M, self._H3, dt=torch.uint8)
        self._sa0a = e(self._M, dt=torch.float32)
        # output must **ping-pong**: the y returned by the previous layer is the next
        # layer's x_in, and the final phase (P6) reads x_in while writing out (the same buffer
        # would silently corrupt the whole layer).
        self._out = [e(self._M, self._H, dt=torch.bfloat16), e(self._M, self._H, dt=torch.bfloat16)]
        self._tog = 0
        # cross-side scratch (kv comes from context, used only inside the kernel, one copy is enough)
        self._ones = torch.ones(self._H, dtype=torch.bfloat16, device=d)
        self._qp = e(self._M, self._H3, dt=torch.bfloat16)
        self._kvp = e(self._C, 6144, dt=torch.bfloat16)
        self._cqc = e(self._M, self._H3, dt=torch.bfloat16)
        self._ckc = e(self._C, self._H3, dt=torch.bfloat16)
        self._cvc = e(self._C, self._H3, dt=torch.bfloat16)
        self._a8q = e(self._M, self._H3, dt=torch.uint8)
        self._sa0q = e(self._M, dt=torch.float32)
        self._a8c = e(self._C, self._H3, dt=torch.uint8)
        self._sa0c = e(self._C, dt=torch.float32)
        self._a8o = e(self._M, self._H3, dt=torch.uint8)
        self._sa0o = e(self._M, dt=torch.float32)
        self._attn2 = e(self._M, self._H3, dt=torch.float32)
        self._rope: dict = {}
        # CUDA graph capture: **no D2H sync at all** during capture (``.item()``/``.tolist()``).
        # grid is preset by ``set_capture_grid()`` before capture (video prefill's (f,h,w) is constant).
        self._capturing = False
        self._capture_grid: tuple | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack_rows(parts, names, what) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .fp8_linear import FP8Linear  # deferred import: avoid module-level circular dependency

        for nm, p in zip(names, parts):
            _require(isinstance(p, FP8Linear), f"{what}.{nm} is not FP8Linear (video fp8-resident required)")
            _require(
                p.weight.dtype == _FP8 and p.weight.is_contiguous(),
                f"{what}.{nm}.weight requires contiguous fp8 e4m3fn",
            )
            _require(
                p.weight_scale.dtype == torch.float32 and p.weight_scale.is_contiguous(),
                f"{what}.{nm} scale requires contiguous fp32",
            )
            _require(
                p.bias is not None and p.bias.dtype == torch.bfloat16 and p.bias.is_contiguous(),
                f"{what}.{nm} requires a contiguous bf16 bias",
            )
        return (
            torch.cat([p.weight.data for p in parts], 0).contiguous(),
            torch.cat([p.weight_scale.data for p in parts], 0).contiguous(),
            torch.cat([p.bias.data for p in parts], 0).contiguous(),
        )

    @classmethod
    def _resolve(cls, blk) -> dict:
        from .fp8_linear import FP8Linear

        sa, ca = blk.self_attn, blk.cross_attn
        if getattr(sa, "qkv", None) is not None:
            wqkv, swqkv, bqkv = sa.qkv.weight, sa.qkv.weight_scale, sa.qkv.bias
        else:
            wqkv, swqkv, bqkv = cls._pack_rows([sa.q, sa.k, sa.v], ("q", "k", "v"), "self_attn")
        if getattr(ca, "kv", None) is not None:
            wkv, swkv, bkv = ca.kv.weight, ca.kv.weight_scale, ca.kv.bias
        else:
            wkv, swkv, bkv = cls._pack_rows([ca.k, ca.v], ("k", "v"), "cross_attn")
        _require(
            isinstance(sa.o, FP8Linear) and isinstance(ca.q, FP8Linear) and isinstance(ca.o, FP8Linear),
            "self.o / cross.q / cross.o must be FP8Linear",
        )
        # normalization weights must be bf16: the engine builds modules under
        # set_default_dtype(bfloat16); if they were fp32, the kernel reading them as bf16
        # would treat the low 16 bits of fp32 as a block of 0 (silently corrupting the whole layer).
        for nm, m in (
            ("self_attn.norm_q", sa.norm_q),
            ("self_attn.norm_k", sa.norm_k),
            ("cross_attn.norm_q", ca.norm_q),
            ("cross_attn.norm_k", ca.norm_k),
            ("norm3", blk.norm3),
        ):
            _require(
                getattr(m, "weight", None) is not None and m.weight.dtype == torch.bfloat16, f"{nm}.weight must be bf16"
            )
        _require(blk.norm3.bias is not None and blk.norm3.bias.dtype == torch.bfloat16, "norm3.bias must be bf16")
        _require(blk.norm1.weight is None, "norm1 must be elementwise_affine=False (Wan default)")
        return dict(
            wqkv=wqkv, swqkv=swqkv, bqkv=bqkv,
            wnq=sa.norm_q.weight.detach().contiguous(),
            wnk=sa.norm_k.weight.detach().contiguous(),
            wo=sa.o.weight.data, swo=sa.o.weight_scale.data, bo=sa.o.bias.data,
            wq=ca.q.weight.data, swq=ca.q.weight_scale.data, bq=ca.q.bias.data,
            wkv=wkv, swkv=swkv, bkv=bkv,
            cwnq=ca.norm_q.weight.detach().contiguous(),
            cwnk=ca.norm_k.weight.detach().contiguous(),
            cwo=ca.o.weight.data, cswo=ca.o.weight_scale.data, cbo=ca.o.bias.data,
            w3=blk.norm3.weight.detach().contiguous(),
            b3=blk.norm3.bias.detach().contiguous(),
        )  # fmt: skip

    def _tables(self, freqs, grid):
        key = (tuple(grid), str(freqs.device))
        t = self._rope.get(key)
        if t is None:
            cos, sin = _video_rope_tables(freqs.detach().cpu(), tuple(grid))
            t = (cos.to(self.device).contiguous(), sin.to(self.device).contiguous())
            self._rope[key] = t
        return t

    def set_capture_grid(self, grid) -> None:
        """Before CUDA graph capture: preset the video rope ``(f, h, w)`` so capture no longer queries grid_sizes."""
        self._capture_grid = tuple(int(v) for v in grid)

    # ------------------------------------------------------------------ #
    def _hook(self, i: int):
        w = self.w[i]

        def hook(
            x,
            context,
            shift_msa,
            scale_msa,
            gate_msa,
            freqs,
            *,
            context_mask=None,
            context_kv=None,
            video_kv=None,
            self_attn_mask=None,
        ):
            _require(
                context_kv is None and video_kv is None,
                "fused attention serves video prefill only (does not accept action-denoise video_kv/context_kv)",
            )
            _require(context is not None, "fused attention requires context (video prefill always provides it)")
            if not self._capturing:
                _require(
                    context_mask is None or bool(context_mask.all()),
                    "the fused path assumes context_mask is all True (which production prefill is)",
                )
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self._H)
            _require(
                x2.shape[0] == self._M,
                f"token count {x2.shape[0]} != kernel compile-time {self._M} (video prefill only supports S=120)",
            )
            _require(x2.is_contiguous(), "x is not contiguous")

            def _mod(t, nm):
                _require(t.shape[-1] == self._H, f"the last dim of {nm} must be hidden")
                if t.dim() < 2:
                    return t, 1, 0
                return t, t.shape[-2], t.stride(-2)

            sh, nh, sr = _mod(shift_msa, "shift_msa")
            sc_, nc, _ = _mod(scale_msa, "scale_msa")
            gt, ng, _ = _mod(gate_msa, "gate_msa")
            _require(
                nh == nc == ng and nh in (1, self._M),
                f"modulation tensors must have equal row counts of 1 or {self._M}",
            )
            mod_srow = 0 if nh == 1 else sr
            # in production they are **views** chunked from `[1,S,6,D]` (row stride 6H), and
            # mod_srow exists precisely to avoid materializing a copy -- so we only require
            # bf16 + contiguous last dim + even row stride.
            for nm, t in (("shift", sh), ("scale", sc_), ("gate", gt)):
                _require(t.dtype == torch.bfloat16, f"{nm}_msa must be bf16")
                _require(t.stride(-1) == 1, f"{nm}_msa last dim must be contiguous")
            _require(nh == 1 or sr % 2 == 0, "modulation row stride must be even (kernel indexes as u32)")

            ob = self._tog
            self._tog ^= 1
            if self._capturing:
                # no .tolist() during capture (D2H sync); grid is preset by set_capture_grid()
                _require(self._capture_grid is not None, "set_capture_grid() is required before graph capture")
                grid = self._capture_grid
            else:
                grid = tuple(int(v) for v in freqs["grid_sizes"].reshape(-1).tolist())
            cos_t, sin_t = self._tables(freqs["freqs"], grid)

            stream = torch.cuda.current_stream().cuda_stream

            self_tensors = dict(
                x_in=x2, shift_msa=sh, scale_msa=sc_, gate_msa=gt,
                wqkv=w["wqkv"], swqkv=w["swqkv"], bqkv=w["bqkv"],
                wnq=w["wnq"], wnk=w["wnk"], cos_t=cos_t, sin_t=sin_t,
                wo=w["wo"], swo=w["swo"], bo=w["bo"],
                out=self._out[ob], qkv=self._qkv, qc=self._qc,
                kc=self._kc[i], vc=self._vc[i], attn=self._attn,
                a8x=self._a8x, sa0x=self._sa0x, a8a=self._a8a, sa0a=self._sa0a,
            )  # fmt: skip
            self_scalars = dict(norm_eps=self.eps, mod_srow=mod_srow, stop_phase=0, dbg=0, phase_ts=0)
            for ph in range(1, 7) if self._split else (-1,):
                phases.dispatch("vdit.attn_self", ph, tensors=self_tensors, scalars=self_scalars, stream=stream)

            c2 = context.reshape(-1, self._H)
            _require(
                c2.shape[0] == self._C and c2.is_contiguous(),
                f"context must be contiguous [{self._C},{self._H}] (got {tuple(c2.shape)})",
            )

            cross_tensors = dict(
                x_in=self._out[ob], context=c2, w3=w["w3"], b3=w["b3"],
                wq=w["wq"], swq=w["swq"], bq=w["bq"],
                wkv=w["wkv"], swkv=w["swkv"], bkv=w["bkv"],
                wnq=w["cwnq"], wnk=w["cwnk"],
                wo=w["cwo"], swo=w["cswo"], bo=w["cbo"], ones=self._ones,
                out=self._out[ob], qp=self._qp, kvp=self._kvp,
                qc=self._cqc, kc=self._ckc, vc=self._cvc, attn=self._attn2,
                a8q=self._a8q, sa0q=self._sa0q, a8c=self._a8c, sa0c=self._sa0c,
                a8o=self._a8o, sa0o=self._sa0o,
            )  # fmt: skip
            cross_scalars = dict(norm_eps=self.eps, stop_phase=0, dbg=0, phase_ts=0)
            for ph in range(1, 7) if self._split else (-1,):
                phases.dispatch("vdit.attn_cross", ph, tensors=cross_tensors, scalars=cross_scalars, stream=stream)
            y = self._out[ob].reshape(*lead, self._H)
            k = self._kc[i].reshape(*lead, self._H3)
            v = self._vc[i].reshape(*lead, self._H3)
            return y, (k, v)

        return hook

    # ------------------------------------------------------------------ #
    def install(self, blocks) -> None:
        _require(len(blocks) == len(self.w), "block count differs from construction")
        for i, blk in enumerate(blocks):
            blk._fused_attn = self._hook(i)
