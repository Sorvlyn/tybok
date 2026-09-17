"""action-expert fused inference runner (three fused CUDA kernels: ``adit.attn_self``, ``adit.attn_cross``, ``adit.ffn``).

Each action block's layer-step is switched from the engine's per-op torch chain to **3 cooperative launches**:

    self-attn  (adit.attn_self)   norm1+AdaLN(msa) -> packed qkv fp8 GEMM -> qk-norm/RoPE ->
                        bf16 flash (video KV cache + fresh 32 rows) -> o fp8 GEMM +
                        gate_msa residual
    cross-attn (adit.attn_cross)  norm3 -> cross-q fp8 GEMM -> RMS -> masked bf16 flash (cached
                        context k/v) -> cross-o fp8 GEMM + residual
    FFN        (adit.ffn)         norm2 -> AdaLN(mlp) -> up fp8 GEMM -> GELU-tanh -> down fp8
                        GEMM -> gate_mlp residual

Numeric contract:
  - **activations I/O are all bf16** (self/cross attention is bf16 flash, not the engine
    reference's fp32 SDPA); weights fp8 e4m3fn + per-row fp32 scale + bf16 bias -- this
    runner directly reuses the fp8-resident ``FP8Linear`` parameters and does no extra
    dequant/requant;
  - quantization/rounding recipe (amax/448 + software RNE, bf16 RN, bias added in fp32
    after dequant) is the same source as ``FP8Linear``;
  - fixed geometry (kernel compile-time constants, action-expert specific): M=32,
    hidden=1024, attn=3072 (24 heads x 128), ffn=4096; used only for action denoise
    (30 layers x each step); video prefill still goes through
    ``FastWAMAttentionBlock.forward``'s torch path.

Numeric tier: fused vs ref (action-fp8 torch chain) chunk drift ~2% rel-RMS (within the
fp8 noise floor); bf16-flash drift tier, **not bit-aligned**. Default off.

Constraints (enforced by the engine when ``--cu-fused-adit`` is on):
  - action expert fp8-resident (fp8 file direct-load), and self-attn's q/k/v are already
    packed into a single [9216, 1024] ``FP8Linear`` (``pack_attention_qkv_fp8``);
  - action cross-attention uses the cross-step-cached context k/v
    (``precompute_action_context``, i.e. the default ``action_context_cache``); the
    cross-attention kernel does not project k/v;
  - action self-attention is unmasked over the full video+action length (single-frame
    first-frame KV == all video KV).
"""

from __future__ import annotations

import torch

from ..kernels import phases
from .dit_block import FastWAMAttentionBlock
from .fp8_linear import FP8Linear

# kernel compile-time geometry (constants from adit_attn_self.cu / adit_attn_cross.cu / adit_ffn.cu)
# fmt: off
_KERNEL_M = 32          # token rows
_KERNEL_H = 1024        # hidden
_KERNEL_H3 = 3072       # attention projection width (24 x 128)
_KERNEL_F = 4096        # FFN width
# fmt: on
_NHEADS = _KERNEL_H3 // 128
_FP8 = torch.float8_e4m3fn
# rstd fragment slots (RSTD_SLOTS in adit_attn_self.cu / HEADS*M in adit_attn_cross.cu):
# one fp32 slot per (head, row), written by ``adit.attn_self``/P3 and ``adit.attn_cross``/P3,
# reduced in fixed order by each kernel's P4. **Not accumulators, no need to zero.**
# fmt: off
_RSTD_SLOTS_SELF = 2 * _NHEADS * _KERNEL_M   # 1536 (self-attn: sum q^2 and sum k^2, each HEADS*M)
_RSTD_SLOTS_CROSS = _NHEADS * _KERNEL_M      # 768 (cross-attn: only sum q^2)
# fmt: on

_GEOM = {
    "hidden": _KERNEL_H,
    "attn": _KERNEL_H3,
    "ffn": _KERNEL_F,
    "heads": _NHEADS,
    "head_dim": 128,
}  # fmt: skip


def _require(pred: bool, msg: str) -> None:
    if not pred:
        raise ValueError(f"[action-fused] {msg}")


def _contiguous_bf16(x: torch.Tensor) -> torch.Tensor:
    """Contiguous bf16 row-major KV (the fused flash kernel reads K/V by row stride H3)."""
    t = x.to(dtype=torch.bfloat16).contiguous()
    _require(t.is_contiguous() and t.dtype == torch.bfloat16, "KV cache must be contiguous bf16")
    return t


class FusedActionRunner:
    """Run the 30-layer action denoise layer by layer as three cooperative launches (self-attn -> cross-attn -> FFN).

    Weights/scale/bias directly reference the fp8-resident module parameters (no copy,
    no structural change); quantization-product scratch buffers are allocated once and
    reused throughout. Inter-layer x ping-pongs between two bf16 [32,1024] buffers,
    producing no per-layer temporary allocations.
    """

    def __init__(self, blocks: list[FastWAMAttentionBlock], device: torch.device, split: bool = False):
        from ..kernels import load_fused_ext

        self.device = torch.device(device)
        # split=True: non-cooperative form -- each of the three kernels is split into plain
        # per-phase launches (self-attn 6 / cross-attn 6 / FFN 4), not requiring whole-GPU co-residency;
        # intra-phase fusion is unchanged.
        self._split = bool(split)
        self.ext = load_fused_ext()
        self.blocks = list(blocks)
        n_layers = len(self.blocks)
        _require(n_layers > 0, "action expert has no blocks")

        # parameter bundle (resolved once per layer; called after weight load/pack/upload)
        self.w = [self._resolve(blk) for blk in self.blocks]
        # stack each layer's modulation [1,6,1024] bf16 parameter into [L,1,6,1024] (one broadcast add per step)
        mods = torch.stack([blk.modulation.data.detach() for blk in self.blocks])
        self._mod_stack = mods.contiguous()  # [L,1,6,H] bf16
        self._stream = None  # refreshed to the current torch stream each step

        # inter-layer x triple-buffer rotation (self-attn -> cross-attn -> FFN serial on the same stream, no in-place writes; caller tokens untouched)
        self._bufs = [torch.empty(_KERNEL_M, _KERNEL_H, dtype=torch.bfloat16, device=self.device) for _ in range(3)]

        # self-attn scratch
        self._s_qkv = torch.empty(_KERNEL_M, 3 * _KERNEL_H3, dtype=torch.bfloat16, device=self.device)
        self._s_attn = torch.empty(_KERNEL_M, _KERNEL_H3, dtype=torch.bfloat16, device=self.device)
        self._s_a8x = torch.empty(_KERNEL_M, _KERNEL_H, dtype=torch.uint8, device=self.device)
        self._s_sa0x = torch.empty(_KERNEL_M, dtype=torch.float32, device=self.device)
        self._s_a8a = torch.empty(_KERNEL_M, _KERNEL_H3, dtype=torch.uint8, device=self.device)
        self._s_sa0a = torch.empty(_KERNEL_M, dtype=torch.float32, device=self.device)
        self._s_rstd = torch.empty(_RSTD_SLOTS_SELF, dtype=torch.float32, device=self.device)
        # cross-attn scratch
        self._g_qp = torch.empty(_KERNEL_M, _KERNEL_H3, dtype=torch.bfloat16, device=self.device)
        self._g_a8q = torch.empty(_KERNEL_M, _KERNEL_H, dtype=torch.uint8, device=self.device)
        self._g_sa0q = torch.empty(_KERNEL_M, dtype=torch.float32, device=self.device)
        self._g_attn = torch.empty(_KERNEL_M, _KERNEL_H3, dtype=torch.bfloat16, device=self.device)
        self._g_a8o = torch.empty(_KERNEL_M, _KERNEL_H3, dtype=torch.uint8, device=self.device)
        self._g_sa0o = torch.empty(_KERNEL_M, dtype=torch.float32, device=self.device)
        self._g_rstd = torch.empty(_RSTD_SLOTS_CROSS, dtype=torch.float32, device=self.device)
        # FFN scratch
        self._f_a8 = torch.empty(_KERNEL_M, _KERNEL_H, dtype=torch.uint8, device=self.device)
        self._f_sa0 = torch.empty(_KERNEL_M, dtype=torch.float32, device=self.device)
        self._f_gbuf = torch.empty(_KERNEL_M, _KERNEL_F, dtype=torch.bfloat16, device=self.device)
        self._f_a8g = torch.empty(_KERNEL_M, _KERNEL_F, dtype=torch.uint8, device=self.device)
        self._f_raw = torch.empty(_KERNEL_M, dtype=torch.uint32, device=self.device)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve(blk: FastWAMAttentionBlock) -> dict[str, torch.Tensor]:
        """Fetch one block's fp8 weights / scale / bias and norm parameters (all references, no copies)."""
        sa, ca = blk.self_attn, blk.cross_attn
        _require(
            isinstance(sa.qkv, FP8Linear),
            "self-attn q/k/v not packed into FP8Linear (requires --pack-qkv or --cu-fused-adit auto-packing)",
        )
        for name, m in (
            ("sa.qkv", sa.qkv),
            ("sa.o", sa.o),
            ("ca.q", ca.q),
            ("ca.o", ca.o),
            ("ffn.0", blk.ffn[0]),
            ("ffn.2", blk.ffn[2]),
        ):
            _require(
                isinstance(m, FP8Linear),
                f"{name} is not FP8Linear (--cu-fused-adit requires "
                "action expert fp8-resident, see engine constraints)",
            )
            _require(m.weight.dtype == _FP8, f"{name}.weight is not fp8 e4m3fn")
            _require(m.weight.is_contiguous(), f"{name}.weight is not contiguous")
        for name, w in (
            ("sa.wnq", sa.norm_q.weight),
            ("sa.wnk", sa.norm_k.weight),
            ("ca.wnq", ca.norm_q.weight),
            ("n3.w", blk.norm3.weight),
            ("n3.b", blk.norm3.bias),
        ):
            _require(
                w is not None and w.dtype == torch.bfloat16 and w.is_contiguous(), f"{name} is missing a bf16 weight"
            )
        _require(blk.norm3.bias is not None, "norm3 requires an affine bias (consistent with checkpoint)")
        _require(
            blk.self_attn.qkv.out_features == 3 * _KERNEL_H3 and blk.self_attn.qkv.in_features == _KERNEL_H,
            "qkv geometry mismatch",
        )
        _require(
            blk.ffn[0].out_features == _KERNEL_F and blk.ffn[2].in_features == _KERNEL_F,
            "FFN geometry mismatch (kernel compile-time fixed 4096)",
        )
        _require(ca.q.in_features == _KERNEL_H and ca.q.out_features == _KERNEL_H3, "cross-q geometry mismatch")
        for m in (sa.o, ca.o):
            _require(
                m.in_features == _KERNEL_H3 and m.out_features == _KERNEL_H,
                "attn o geometry mismatch (kernel expects [1024, 3072] layout)",
            )
        return dict(
            # self-attn
            wqkv=sa.qkv.weight, swqkv=sa.qkv.weight_scale, bqkv=sa.qkv.bias,
            wnq=sa.norm_q.weight, wnk=sa.norm_k.weight,
            wo=sa.o.weight, swo=sa.o.weight_scale, bo=sa.o.bias,
            # cross-attn
            w3=blk.norm3.weight, b3=blk.norm3.bias,
            wq=ca.q.weight, swq=ca.q.weight_scale, bq=ca.q.bias,
            wnq_c=ca.norm_q.weight,
            wo_c=ca.o.weight, swo_c=ca.o.weight_scale, bo_c=ca.o.bias,
            # FFN
            w0=blk.ffn[0].weight, sw0=blk.ffn[0].weight_scale, b0=blk.ffn[0].bias,
            w1=blk.ffn[2].weight, sw1=blk.ffn[2].weight_scale, b1=blk.ffn[2].bias,
        )  # fmt: skip

    # ------------------------------------------------------------------ #
    # per denoise step: one call for the whole action chain (30 layers)
    # ------------------------------------------------------------------ #
    def prepare(self, t_mod: torch.Tensor, freqs: torch.Tensor, context_mask_row: torch.Tensor) -> tuple:
        """Once per step: shared quantities (modsum / RoPE tables / mask row)."""
        self._stream = torch.cuda.current_stream().cuda_stream
        modsum = self._mod_stack + t_mod.view(1, 1, 6, _KERNEL_H)  # bf16
        f64 = freqs.reshape(-1, _KERNEL_H3 // _NHEADS // 2).to(torch.complex64)
        cos_t = f64.real.contiguous()
        sin_t = f64.imag.contiguous()
        _require(cos_t.shape == (_KERNEL_M, 64), f"freqs geometry mismatch {tuple(cos_t.shape)}")
        mask = context_mask_row.reshape(-1).contiguous()
        return modsum, cos_t, sin_t, mask

    def step_layer(
        self,
        layer_idx: int,
        src: torch.Tensor,
        prep: tuple,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        L_video: int,
        L_context: int,
    ) -> torch.Tensor:
        """Single-layer self-attn -> cross-attn -> FFN (src -> triple-buffer rotation slot), returns this layer's output [32, H] 2D.

        ``src`` must be contiguous bf16 [32, 1024] (or [1, 32, 1024], which is flattened),
        and must not alias an internal rotation buffer (the caller guarantees rotation
        order for the whole-layer chain)."""
        if src.ndim == 3:
            _require(tuple(src.shape) == (1, _KERNEL_M, _KERNEL_H), f"src shape {tuple(src.shape)} != [1, 32, 1024]")
            src = src.view(_KERNEL_M, _KERNEL_H)
        _require(
            src.dtype == torch.bfloat16 and src.is_cuda and src.is_contiguous(), "src must be contiguous bf16 CUDA"
        )
        modsum, cos_t, sin_t, mask = prep
        buf0, buf1, buf2 = self._bufs
        i = layer_idx
        s_out, g_out, f_out = (
            (buf0, buf1, buf2) if i % 3 == 0 else ((buf1, buf2, buf0) if i % 3 == 1 else (buf2, buf0, buf1))
        )
        w = self.w[i]
        mod = modsum[i].chunk(6, dim=1)
        (sh_m, sc_m, gt_m, sh_f, sc_f, gt_f) = (c.view(1, _KERNEL_H) for c in mod)
        stream = self._stream
        # non-cooperative form: each kernel is split into plain per-phase launches
        # (self-attn 6 / cross-attn 6 / FFN 4), ordered between phases by stream order; the cooperative form
        # (split=False) is still 1 cooperative launch per kernel. only_phase=-1 = full pipeline.
        self_phases = range(1, 7) if self._split else (-1,)
        cross_phases = range(1, 7) if self._split else (-1,)
        ffn_phases = range(1, 5) if self._split else (-1,)
        # ---- self-attn: full chain (src -> s_out) ----
        s_tensors = dict(
            x_in=src, shift_msa=sh_m, scale_msa=sc_m,
            wqkv=w["wqkv"], swqkv=w["swqkv"], bqkv=w["bqkv"],
            kv_cache=video_k, v_cache=video_v, wnq=w["wnq"], wnk=w["wnk"],
            cos_t=cos_t, sin_t=sin_t, gate_msa=gt_m,
            wo=w["wo"], swo=w["swo"], bo=w["bo"],
            out=s_out, qkv=self._s_qkv, attn=self._s_attn,
            a8x=self._s_a8x, sa0x=self._s_sa0x, a8a=self._s_a8a, sa0a=self._s_sa0a,
            rstd_scratch=self._s_rstd,
        )  # fmt: skip
        for ph in self_phases:
            phases.dispatch("adit.attn_self", ph, tensors=s_tensors, scalars=dict(L=L_video), stream=stream)
        # ---- cross-attn: full path (s_out -> g_out) ----
        g_tensors = dict(
            x_in=s_out, w3=w["w3"], b3=w["b3"],
            wq=w["wq"], swq=w["swq"], bq=w["bq"],
            qp=self._g_qp, a8q=self._g_a8q, sa0q=self._g_sa0q, wnq=w["wnq_c"],
            k_cache=cross_k, v_cache=cross_v, mask=mask, attn=self._g_attn,
            wo=w["wo_c"], swo=w["swo_c"], bo=w["bo_c"], out=g_out,
            a8o=self._g_a8o, sa0o=self._g_sa0o, rstd_scratch=self._g_rstd,
        )  # fmt: skip
        for ph in cross_phases:
            phases.dispatch("adit.attn_cross", ph, tensors=g_tensors, scalars=dict(L=L_context), stream=stream)
        # ---- FFN: g_out -> f_out (the next layer's self-attn reads from f_out) ----
        f_tensors = dict(
            x_in=g_out, shift_mlp=sh_f, scale_mlp=sc_f, gate_mlp=gt_f,
            w0=w["w0"], sw0=w["sw0"], b0=w["b0"],
            w1=w["w1"], sw1=w["sw1"], b1=w["b1"],
            out=f_out, a8buf=self._f_a8, sa0buf=self._f_sa0,
            gbuf=self._f_gbuf, a8gbuf=self._f_a8g, raw1=self._f_raw,
        )  # fmt: skip
        for ph in ffn_phases:
            phases.dispatch("adit.ffn", ph, tensors=f_tensors, scalars=dict(B=1), stream=stream)
        return f_out

    def step_layer_prepared(
        self,
        layer_idx: int,
        src: torch.Tensor,
        prep: tuple,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
    ) -> torch.Tensor:
        """``step_layer`` with the contiguous-bf16 KV normalization that
        ``run_action_layers`` applies (per-layer form, for the overlap branch:
        the video KV is only available once prefill layer ``i`` has run).

        ``prep`` comes from :meth:`prepare` and must have been built on the stream
        the layers will run on (kernels launch on the stream recorded there).
        """
        k = _contiguous_bf16(video_k).reshape(1, -1, _KERNEL_H3)
        v = _contiguous_bf16(video_v).reshape(1, -1, _KERNEL_H3)
        kc = _contiguous_bf16(cross_k).reshape(1, -1, _KERNEL_H3)
        vc = _contiguous_bf16(cross_v).reshape(1, -1, _KERNEL_H3)
        return self.step_layer(layer_idx, src, prep, k, v, kc, vc, k.shape[1], kc.shape[1])

    # Parameter table: the aligned trailing comments name each argument's dtype/shape. The
    # signature carries `# fmt: skip`; the return annotation that used to sit on the closing
    # line is already stated in the docstring below.
    def run_action_layers(
        self,
        tokens: torch.Tensor,            # bf16 [1, M=32, H=1024]
        t_mod: torch.Tensor,             # bf16 [1, 6, H]
        freqs: torch.Tensor,             # complex [M, 1, d/2] (action dense rope tables)
        video_kv: list[dict],            # per layer {"k","v"} bf16 [1, L_video, 3072]
        cross_kv: list[tuple],           # per layer (k, v) bf16 [1, L, 3072] (qk-norm applied)
        context_mask: torch.Tensor,      # bool [L] (same for every row, invariant across layers)
    ) -> torch.Tensor:  # fmt: skip
        """Run all 30 layers of self-attn -> cross-attn -> FFN, return bf16 [1, 32, 1024] (for the action head)."""
        x = self._check_and_flatten(tokens)
        n_layers = len(self.blocks)
        _require(
            len(video_kv) == n_layers and len(cross_kv) == n_layers,
            f"video/cross KV layer counts {len(video_kv)}/{len(cross_kv)} != {n_layers}",
        )
        # defensive: the fused kernel reads k/v as contiguous [L_video|L, H3] (cp.async by
        # row stride H3). prefill's v and cross's k/v are often **strided slices** of the
        # packed projection (row stride 3*H3), and passing a non-contiguous view directly
        # would make flash read misaligned. Copy once per chunk.
        video_kv = [
            {
                "k": _contiguous_bf16(c["k"]).reshape(1, -1, _KERNEL_H3),
                "v": _contiguous_bf16(c["v"]).reshape(1, -1, _KERNEL_H3),
            }
            for c in video_kv
        ]
        cross_kv = [
            (_contiguous_bf16(k).reshape(1, -1, _KERNEL_H3), _contiguous_bf16(v).reshape(1, -1, _KERNEL_H3))
            for k, v in cross_kv
        ]
        L_video = video_kv[0]["k"].shape[1]
        L_context = cross_kv[0][0].shape[1]
        _require(context_mask.numel() == L_context, "context mask length != context KV row count")
        prep = self.prepare(t_mod, freqs, context_mask)
        for i in range(n_layers):
            x = self.step_layer(
                i, x, prep, video_kv[i]["k"], video_kv[i]["v"], cross_kv[i][0], cross_kv[i][1], L_video, L_context
            )
        return x.view(1, _KERNEL_M, _KERNEL_H)

    def _check_and_flatten(self, tokens: torch.Tensor) -> torch.Tensor:
        _require(
            tokens.dtype == torch.bfloat16 and tokens.is_cuda and tokens.is_contiguous(),
            "tokens must be contiguous bf16 CUDA",
        )
        _require(
            tuple(tokens.shape) == (1, _KERNEL_M, _KERNEL_H),
            f"action tokens shape {tuple(tokens.shape)} != [1, 32, 1024] (the fused kernel uses fixed geometry)",
        )
        return tokens.view(_KERNEL_M, _KERNEL_H)
