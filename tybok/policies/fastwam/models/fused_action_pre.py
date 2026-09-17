"""ActionDiT pre-component fusion runner (``--action-pre-fused``).

- ``tmod(timestep)``: replaces `sinusoidal_embedding_1d -> time_embedding ->
  time_projection` in `ActionDiT.pre_dit`, collapsed into 4 small launches.
- ``precompute_ctx(context, context_mask)``: replaces `precompute_action_context`
  (30 layers x one FP8Linear + norm_k for cross kv each, re-quantizing the same
  context_emb 30 times), collapsed into "text_embedding + 1 quantization + 1 big
  GEMM + 1 repack".

Numerics: the context path is bit-identical to production; the time path differs
only in fp32 reduction order (sub-ulp in bf16).
"""

from __future__ import annotations

import torch

from ..kernels import load_action_pre_ext
from .fp8_linear import fp8_gemm, quantize_act

# fmt: off
_H = 1024          # action hidden
_NF = 256          # freq dim
_OUT = 6144        # 6 * hidden
_H3 = 3072         # attention dim (24 x 128)
# fmt: on


def _t_fp8(w: torch.Tensor) -> torch.Tensor:
    """fp8 [N,K] -> [K,N] contiguous (fp8 .t().contiguous() may be unsupported on old torch)."""
    try:
        return w.t().contiguous()
    except Exception:  # noqa: BLE001 - fall back to uint8 view transpose
        u = w.view(torch.uint8).t().contiguous()
        return u.view(torch.float8_e4m3fn)


class ActionPreRunner:
    def __init__(self, action_expert, device: torch.device):
        self._quantize_act = quantize_act
        self._fp8_gemm = fp8_gemm
        self.device = torch.device(device)
        self.ext = load_action_pre_ext()

        time_embed, time_proj = action_expert.time_embedding, action_expert.time_projection
        # weight transposed [K,N]; scale [N]; bias bf16 [N]
        self.w0t = _t_fp8(time_embed[0].weight.data)
        self.sw0 = time_embed[0].weight_scale.data.contiguous()
        self.b0 = time_embed[0].bias.data.contiguous()
        self.w1t = _t_fp8(time_embed[2].weight.data)
        self.sw1 = time_embed[2].weight_scale.data.contiguous()
        self.b1 = time_embed[2].bias.data.contiguous()
        self.w2t = _t_fp8(time_proj[1].weight.data)
        self.sw2 = time_proj[1].weight_scale.data.contiguous()
        self.b2 = time_proj[1].bias.data.contiguous()

        # freq table (fp64, same formula as wan_base.sinusoidal_embedding_1d)
        half = _NF // 2
        self.freq = torch.pow(10000, -torch.arange(half, dtype=torch.float64, device=self.device) / half).contiguous()

        self.y0 = torch.empty(_H, dtype=torch.bfloat16, device=self.device)
        self.y1 = torch.empty(_H, dtype=torch.bfloat16, device=self.device)
        self.y2 = torch.empty(_H, dtype=torch.bfloat16, device=self.device)
        self.amax = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._tmod_buf = torch.empty(_OUT, dtype=torch.bfloat16, device=self.device)
        self.ts = torch.empty(1, dtype=torch.float32, device=self.device)

        # ---- context side: stack 30 layers of cross kv weights + norm_k ----
        blocks = list(action_expert.blocks)
        self.nb = len(blocks)
        kvs, wnks = [], []
        for blk in blocks:
            cross_attn = blk.cross_attn
            if getattr(cross_attn, "kv", None) is not None:
                kvs.append(cross_attn.kv)
            else:  # not packed: concatenate k/v rows
                kvs.append(None)
            wnks.append(cross_attn.norm_k.weight.data)
        if any(k is None for k in kvs):
            raise NotImplementedError("action_pre requires cross_attn.kv to be packed (--pack-qkv / --cu-fused-adit)")
        self.Wkv = torch.cat([m.weight.data for m in kvs], dim=0).contiguous()  # [NB*6144,1024]
        self.swkv = torch.cat([m.weight_scale.data for m in kvs], dim=0).contiguous()
        self.bkv = torch.cat([m.bias.data for m in kvs], dim=0).contiguous()
        self.wnk = torch.cat(wnks, dim=0).contiguous()  # [NB,3072]
        self.eps = float(blocks[0].cross_attn.norm_k.eps)
        self.text_embedding = action_expert.text_embedding

        self._out_big = None
        self._kv_out = None

    # ------------------------------------------------------------------ #
    def tmod(self, timestep: torch.Tensor) -> torch.Tensor:
        """timestep bf16 [1] -> t_mod bf16 [1,6,H]."""
        self.ts.copy_(timestep.reshape(1).to(torch.float32))
        self.ext.ap_time_path(
            self.ts.data_ptr(), self.freq.data_ptr(),
            self.w0t.data_ptr(), self.sw0.data_ptr(), self.b0.data_ptr(),
            self.w1t.data_ptr(), self.sw1.data_ptr(), self.b1.data_ptr(),
            self.w2t.data_ptr(), self.sw2.data_ptr(), self.b2.data_ptr(),
            self.y0.data_ptr(), self.y1.data_ptr(), self.y2.data_ptr(),
            self.amax.data_ptr(), (self.amax.data_ptr() + 4), (self.amax.data_ptr() + 8),
            self._tmod_buf.data_ptr(), torch.cuda.current_stream().cuda_stream)  # fmt: skip
        return self._tmod_buf.view(1, 6, _H)

    # ------------------------------------------------------------------ #
    def precompute_ctx(self, context: torch.Tensor, context_mask=None) -> dict:
        """Same return as `ActionDiT.precompute_action_context` (emb / kv / mask)."""
        context = context.to(dtype=torch.bfloat16)
        context_emb = self.text_embedding(context)  # 2x FP8Linear + GELU
        L = context_emb.shape[0] * context_emb.shape[1]
        emb2 = context_emb.reshape(L, _H).contiguous()
        a8, sa = self._quantize_act(emb2)  # quantize once (production quantizes 30 times)
        if self._out_big is None or self._out_big.shape[0] != L:
            self._out_big = torch.empty(L, self.nb * _OUT, dtype=torch.bfloat16, device=self.device)
            self._kv_out = torch.empty(self.nb, 2, L, _H3, dtype=torch.bfloat16, device=self.device)
        self._fp8_gemm(a8, self.Wkv, sa, self.swkv, self.bkv, self._out_big, cfg=(64, 64, 128, 4, 3))
        self.ext.ap_ctx_repack(self._out_big.data_ptr(), L, self.nb,
                               self.wnk.data_ptr(), self.eps, self._kv_out.data_ptr(),
                               torch.cuda.current_stream().cuda_stream)  # fmt: skip
        B = context.shape[0]
        q = L // B
        kv = [(self._kv_out[l, 0].view(B, q, _H3), self._kv_out[l, 1].view(B, q, _H3)) for l in range(self.nb)]
        return {"emb": context_emb, "kv": kv, "mask": context_mask}

    # ------------------------------------------------------------------ #
    def install(self, action_expert) -> None:
        action_expert._fused_tmod = self.tmod
        action_expert._fused_ctx_pre = self.precompute_ctx
