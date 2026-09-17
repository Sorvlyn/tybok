# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 Physical
# Intelligence and The HuggingFace Inc. team; Copyright 2026 The HuggingFace Inc. team. Licensed
# under the Apache License, Version 2.0; modified for TyBoK in 2026. See the NOTICE file.

"""PI05FlowMatching: prefix embedding + flow-matching denoising loop.

Entry point :meth:`PI05FlowMatching.sample_actions`: ``embed_prefix`` builds the
prefix KV cache, then ``euler_integrate`` runs the Euler loop against it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
import triton
import triton.language as tl
from torch import Tensor, nn

from ..config import PI05Config
from .paligemma_with_expert import PaliGemmaWithExpertModel

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# --------------------------------------------------------------------------- #
# fused final AdaRMS norm + action_out_proj (``fuse_final_tail``)
# --------------------------------------------------------------------------- #
# Replaces ``expert.norm -> [to fp32] -> action_out_proj`` with one kernel: per-row fp32
# variance, then ``round_bf16(x*rstd*(1+scale)+shift)`` -- the exact eager rounding point, so
# the result must stay bit-exact vs eager -- dotted in true fp32 (``ieee``, not TF32) with an
# fp32 out. Final modulation is scale/shift-only ``[2*K]`` (no gated residual follows).

# transposed [K, N] fp32 out weight cache (data_ptr-keyed + strong ref)
_final_wt_cache: dict[int, tuple[Tensor, Tensor]] = {}


# Launch-parameter table: the aligned trailing comments name each argument's shape. The signature
# carries ``# fmt: skip`` so the formatter leaves the table alone; the body is formatted.
@triton.jit
def _final_norm_out_kernel(
    x_ptr,          # [M, K] bf16 pre-norm suffix hidden (row-major)
    mod_ptr,        # [2*K] f32 final-norm modulation (scale, shift; no gate)
    wt_ptr,         # [K, N_pad] fp32 transposed action_out_proj weight
    out_ptr,        # [M, N] fp32 velocity out
    M,
    K,
    N,
    EPS,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):  # fmt: skip
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < M

    # AdaRMS variance pass over the full K (fp32), the eager GemmaRMSNorm recipe
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
        )
        xf = xv.to(tl.float32)
        sum_sq += tl.sum(xf * xf, 1)
    rstd = tl.rsqrt(sum_sq / K + EPS)

    # second pass: single-rounding normed bf16 tiles -> true-fp32 dot with W
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
        )
        xf = xv.to(tl.float32)
        sc = tl.load(mod_ptr + (k0 + offs_k))
        sh = tl.load(mod_ptr + K + (k0 + offs_k))
        # round_bf16 then back to fp32 (exact): the values the eager tail feeds its fp32 matmul
        a = (xf * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]).to(tl.bfloat16).to(tl.float32)
        wt = tl.load(
            wt_ptr + (k0 + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_n < N)[None, :],
            other=0.0,
        )
        acc = tl.dot(a, wt, acc=acc, input_precision="ieee")
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=m_mask[:, None] & (offs_n < N)[None, :],
    )


def _final_out_weight_t(w: Tensor) -> Tensor:
    """[N, K] fp32 action_out_proj weight -> cached contiguous [K, N] (row-major)."""
    ptr = w.data_ptr()
    e = _final_wt_cache.get(ptr)
    if e is None:
        t = w.transpose(0, 1).contiguous()
        if len(_final_wt_cache) > 16:
            _final_wt_cache.clear()
        _final_wt_cache[ptr] = (t, w)
        return t
    return e[0]


def final_norm_out_proj(hidden: Tensor, modulation: Tensor, eps: float, out_proj: nn.Module) -> Tensor:
    """Fused final AdaRMS norm + action_out_proj.

    ``hidden`` is the ``[1, M, K]`` bf16 *pre-norm* suffix hidden, ``modulation`` the
    final-norm ``dense(cond)`` scale/shift ``[1, 2*K]`` f32 (no gate); returns ``[1, M, N]`` fp32.
    """
    b, m, k = hidden.shape
    n = out_proj.out_features
    block_n = triton.next_power_of_2(n)
    wt = _final_out_weight_t(out_proj.weight.data)
    out = torch.empty(b, m, n, dtype=torch.float32, device=hidden.device)
    _final_norm_out_kernel[(triton.cdiv(m, 64),)](
        hidden,
        modulation.reshape(-1),
        wt,
        out,
        m,
        k,
        n,
        float(eps),
        hidden.stride(1),
        hidden.stride(2),
        wt.stride(0),
        wt.stride(1),
        out.stride(1),
        out.stride(2),
        BLOCK_M=64,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=2,
    )
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Sine-cosine positional embedding for scalar positions (openpi convention)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float64 if "cuda" in str(device) or "cpu" in str(device) else time.dtype
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """big_vision-style 2-D boolean attention mask."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: Tensor) -> Tensor:
    """Boolean 2-D mask -> additive 4-D mask (0.0 / OPENPI_ATTENTION_MASK_VALUE)."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


def sample_noise(shape, device, generator=None) -> Tensor:
    """Standard-normal float32 noise, the flow-matching x_1 sample."""
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device, generator=generator)


def euler_integrate(
    denoise_fn,
    noise: Tensor,
    num_steps: int,
    *,
    rtc_processor=None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
    time_tensors: list[Tensor] | None = None,
    time_embs: list[Tensor] | None = None,
    adarms_mods: list | None = None,
    adarms_conds: list[Tensor] | None = None,
    start_step: int = 0,
) -> Tensor:
    """Forward-Euler integration from t=1 (noise) to t=0 (actions).

    ``dt = -1/num_steps``, ``time = 1.0 + step*dt``, ``x_t <- x_t + dt * v_t``, with the
    optional real-time-chunking (RTC) guidance hook wrapping the velocity computation.

    ``time_tensors`` / ``time_embs`` / ``adarms_mods`` / ``adarms_conds`` are the precomputed
    per-step constants of the CUDA-graph path: they replace per-step ``torch.tensor``
    host->device copies (illegal while capturing) and must equal the eager values bit-exactly.
    """
    bsize = noise.shape[0]
    device = noise.device
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(start_step, num_steps):
        time = 1.0 + step * dt
        if time_tensors is not None:
            time_tensor = time_tensors[step]
        else:
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        time_emb = time_embs[step] if time_embs is not None else None
        mods = adarms_mods[step] if adarms_mods is not None else None
        cond = adarms_conds[step] if adarms_conds is not None else None

        def denoise_step_partial_call(
            input_x_t, current_timestep=time_tensor, _time_emb=time_emb, _mods=mods, _cond=cond
        ):
            return denoise_fn(input_x_t, current_timestep, _time_emb, _mods, _cond)

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=execution_horizon,
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        x_t = x_t + dt * v_t

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t


# --------------------------------------------------------------------------- #
# PI05FlowMatching
# --------------------------------------------------------------------------- #
class PI05FlowMatching(nn.Module):
    def __init__(self, config: PI05Config, rtc_processor=None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self.paligemma_with_expert = PaliGemmaWithExpertModel(config)
        expert_width = config.expert.width

        self.action_in_proj = nn.Linear(config.max_action_dim, expert_width, dtype=torch.float32)
        self.action_out_proj = nn.Linear(expert_width, config.max_action_dim, dtype=torch.float32)

        self.time_mlp_in = nn.Linear(expert_width, expert_width, dtype=torch.float32)
        self.time_mlp_out = nn.Linear(expert_width, expert_width, dtype=torch.float32)

        # Engine-injected RNG for the denoising noise (--seed); None -> the global torch RNG.
        self.noise_generator: torch.Generator | None = None

        # --skip-empty-cams: an empty camera slot is pad-masked out of every attention, so its
        # ViT+projector pass is dead work; skip it and use a same-shape zeros placeholder.
        self.skip_empty_images = False

        # --pad-free: compact the prefix to live tokens only (empty-camera blocks and language
        # pads are already masked out of every attention). Eager-only: the live count is a
        # GPU->host sync (``nonzero()``), illegal in a captured graph. The changed score-GEMM
        # shape may also change cuBLAS accumulation order -> drift; see ``pack_language``.
        self.pad_free = False

        # --pad-free (graph path): number of image slots the graph captures (engine-set before
        # capture; None = ``len(image_features)``). Sizes the static prefix masks.
        self._n_img_slots: int | None = None

        # --pad-free (graph path): bucketed language length the engine sets before capture so the
        # static prefix masks and the language block are sized to ``lang_len``; None = no
        # compaction.
        self._lang_len: int | None = None

        # CUDA-graph static path: precomputed constants replacing the per-call host->device
        # tensor construction / recomputation, which is illegal inside a captured graph.
        # Bit-identical to the eager path. Enabled by the engine before graph capture.
        self.use_static = False
        self._time_tensors: list[Tensor] | None = None
        self._time_embs: list[Tensor] | None = None
        self._suffix_att_masks: Tensor | None = None
        self._prefix_att_masks: Tensor | None = None
        # Per-step, per-layer AdaRMS modulations ``dense(time_mlp(time_emb))``, hoisted out of
        # the denoising loop (the conditioning depends only on the timestep):
        # ``_adarms_mods[step]`` is ``[(in_mod, post_mod)] * n_layers + [(norm_ss, None)]`` --
        # the last entry is scale/shift-only, its gate is never consumed.
        self._adarms_mods: list[list[tuple[Tensor, Tensor | None]]] | None = None
        self._adarms_cond: list[Tensor] | None = None

        # ``prefill_layer`` x ``step0_layer`` fusion (engine-set, CUDA only): prefill layer ``i``
        # runs on the main stream while step-0 expert layer ``i`` runs on a side stream gated on
        # prefix KV[i]. Bit-identical to the sequential path; ``--graph`` captures it as one graph.
        self.overlap = False
        self._overlap_stream: torch.cuda.Stream | None = None
        self._overlap_events: list[torch.cuda.Event] | None = None

    # ------------------------------------------------------------------ #
    # static (CUDA-graph) constant precomputation
    # ------------------------------------------------------------------ #
    def _precompute_constants(self, device: str) -> None:
        """Precompute every loop-invariant constant for the captured graph.

        Per-step timestep scalars and sinusoidal embeddings (bit-identical to the eager
        ``embed_suffix`` / ``compute_adarms_cond``), the suffix/prefix attention masks, and the
        per-step AdaRMS modulations. Allocates and copies host->device, so never call it
        while capturing.
        """
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        self._time_tensors = [
            torch.tensor([1.0 + step * dt], dtype=torch.float32, device=device) for step in range(num_steps)
        ]
        times = torch.cat([t for t in self._time_tensors], dim=0)
        time_embs = create_sinusoidal_pos_embedding(
            times,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        # the eager ``embed_suffix`` casts the fp64 result to the timestep dtype (fp32); same here
        time_embs = time_embs.type(dtype=torch.float32)
        self._time_embs = [time_embs[i : i + 1].contiguous() for i in range(num_steps)]
        att_masks = [1.0] + [0.0] * (self.config.chunk_size - 1)
        self._suffix_att_masks = torch.tensor(att_masks, dtype=torch.float32, device=device)[None, :]
        n_img = self._n_img_slots if self._n_img_slots is not None else len(self.config.image_features)
        lang_len = self._lang_len if self._lang_len is not None else self.config.tokenizer_max_length
        prefix_len = 256 * n_img + lang_len
        self._prefix_att_masks = torch.zeros(1, prefix_len, dtype=torch.bool, device=device)

        # AdaRMS modulations: ``dense(time_mlp(time_emb))`` per step for every expert norm
        # (input/post per layer + the final norm). Bit-identical to the eager path.
        conds: list[Tensor] = []
        for step in range(num_steps):
            conds.append(self.compute_adarms_cond(self._time_embs[step]))
        self._adarms_cond = conds
        self._adarms_mods = [self.compute_adarms_mods(cond) for cond in conds]

    def constant_tensors(self) -> tuple:
        """Every tensor :meth:`_precompute_constants` allocated, for keepalive.

        Captured graphs read these buffers by address and they are model-level, so a later
        ``_precompute_constants`` (new camera count or language bucket) would replace them:
        the engine snapshots this tuple into each graph entry to keep them alive.
        """
        return (
            self._time_tensors,
            self._time_embs,
            self._suffix_att_masks,
            self._prefix_att_masks,
            self._adarms_cond,
            self._adarms_mods,
        )

    def compute_adarms_cond(self, time_emb: Tensor) -> Tensor:
        """Build the AdaRMS conditioning from the time embedding (bit-identical to eager)."""
        cond = self.time_mlp_in(time_emb)
        cond = F.silu(cond)
        cond = self.time_mlp_out(cond)
        return F.silu(cond)

    def compute_adarms_mods(self, cond: Tensor) -> list[tuple[Tensor, Tensor | None]]:
        """Build the per-norm ``dense(cond)`` modulations the fused layers consume.

        Returns ``[(input, post)] * n_layers + [(final_norm_scale_shift, None)]``; the last
        entry is scale/shift-only ``[1, 2*width]`` because no gated residual follows the final
        norm, so its gate segment is not projected.
        """
        expert = self.paligemma_with_expert.gemma_expert.model
        mods: list[tuple[Tensor, Tensor | None]] = []
        for layer in expert.layers:
            mods.append(
                (
                    layer.input_layernorm.dense(cond),
                    layer.post_attention_layernorm.dense(cond),
                )
            )
        width = expert.width
        final_norm_dense = expert.norm.dense
        mods.append(
            (
                F.linear(cond, final_norm_dense.weight[: 2 * width], final_norm_dense.bias[: 2 * width]),
                None,
            )
        )
        return mods

    # ------------------------------------------------------------------ #
    def embed_prefix(self, images, img_masks, tokens, masks) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images + language into the prefix; returns ``(embs, pad_masks, att_masks)``.

        ``embs`` is fp32 via ``torch.cat`` type promotion.
        """
        embs = []
        pad_masks = []
        att_masks = []
        placeholder_shape: tuple[int, int] | None = None

        for img, img_mask in zip(images, img_masks, strict=True):
            # Eager-path only: deciding this needs a GPU->host sync (``img_mask.all()``), which
            # is illegal inside a captured graph. Under ``use_static`` the masks are all-True
            # anyway, and with the flag off this stays bit-identical to the plain path.
            if self.skip_empty_images and not self.use_static and not bool(img_mask.all()):
                if placeholder_shape is not None:
                    # Empty camera (pad-masked out of every attention): skip the ViT+projector
                    # and pad the prefix with zeros of the same shape; positions/masks untouched.
                    bsize = img.shape[0]
                    img_emb = torch.zeros((bsize, *placeholder_shape), dtype=torch.float32, device=img.device)
                    num_img_embs = placeholder_shape[0]
                    embs.append(img_emb)
                    pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                    att_masks += [0] * num_img_embs
                    continue
                # Empty camera before any real one: shape unknown, fall through (safe, just slower).
            img_emb = self.paligemma_with_expert.embed_image(img)
            if placeholder_shape is None:
                placeholder_shape = (img_emb.shape[1], img_emb.shape[2])
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        if self.use_static and self._prefix_att_masks is not None:
            # all-False constant of the fixed prefix length, already ``[1, prefix_len]`` (bsize
            # is fixed to 1); the ``torch.tensor`` below would be an illegal host->device copy.
            att_masks = self._prefix_att_masks
        else:
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
            bsize = pad_masks.shape[0]
            att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self, noisy_actions, timestep, time_emb: Tensor | None = None, adarms_cond: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed the noisy actions + timestep for the expert Gemma (AdaRMS condition).

        ``time_emb`` / ``adarms_cond`` (CUDA-graph path) are the precomputed constants for
        this timestep; ``None`` recomputes them eagerly (identical values).
        """
        att_masks = []

        if time_emb is None:
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=timestep.device,
            )
            time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)

        if adarms_cond is None:
            adarms_cond = self.compute_adarms_cond(time_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)

        # Image/language inputs must not attend to action tokens; actions are causal within
        # the chunk. The mask is the fixed constant ``[1] + [0]*(chunk-1)``; the CUDA-graph
        # path uses the precomputed device tensor instead of a host->device ``torch.tensor``.
        if self.use_static and self._suffix_att_masks is not None:
            att_masks = self._suffix_att_masks
        else:
            att_masks += [1] + ([0] * (self.config.chunk_size - 1))
            att_masks = torch.tensor(att_masks, dtype=action_emb.dtype, device=action_emb.device)
            att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return action_emb, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------ #
    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]],
        x_t: Tensor,
        timestep: Tensor,
        time_emb: Tensor | None = None,
        adarms_mods: list[tuple[Tensor, Tensor | None]] | None = None,
        adarms_cond: Tensor | None = None,
    ) -> Tensor:
        """One denoising step: expert forward over the prefix KV + suffix -> velocity.

        ``time_emb`` / ``adarms_mods`` / ``adarms_cond`` (CUDA-graph path) are the precomputed
        per-step constants (see ``embed_suffix`` / ``_precompute_constants``).
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond_out = self.embed_suffix(
            x_t, timestep, time_emb=time_emb, adarms_cond=adarms_cond
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond_out],
            adarms_mods=[None, adarms_mods],
            triton_prefix_pad=[None, prefix_pad_masks],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        if getattr(self, "fuse_final_tail", False) and adarms_mods is not None:
            # the expert decoder skipped its final norm (``skip_final_norm``): fuse it with
            # the out projection here
            expert = self.paligemma_with_expert.gemma_expert.model
            return final_norm_out_proj(suffix_out, adarms_mods[-1][0], expert.norm.eps, self.action_out_proj)
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    # ------------------------------------------------------------------ #
    # ``prefill_layer`` x ``step0_layer`` fusion (``overlap``)
    #
    # Step-0 expert layer ``i`` cross-attends ``prefix_kv[i]`` only (written when prefill
    # layer ``i`` finishes), so step 0 hides inside the prefill window. It is plain model
    # code, not a graph construct: eager runs it as-is and ``--graph`` captures the whole
    # fused branch as one graph per shape. Bit-identical to the sequential path.
    # ------------------------------------------------------------------ #
    def _finish_step0(self, suffix_hidden, noise, adarms_cond, adarms_mods, dt) -> Tensor:
        """Step-0 tail: expert final norm + velocity projection + Euler update.

        The tail of ``denoise_step`` for step 0, hoisted so the fused branch can run it once
        after the interleaved layer loops (the final norm sits outside the layer loop).
        """
        expert = self.paligemma_with_expert.gemma_expert.model
        suffix_out = suffix_hidden[:, -self.config.chunk_size :]
        if getattr(self, "fuse_final_tail", False) and adarms_mods is not None:
            v0 = final_norm_out_proj(suffix_out, adarms_mods[-1][0], expert.norm.eps, self.action_out_proj)
        else:
            hidden, _ = expert.norm(suffix_hidden, adarms_cond, adarms_mods[-1][0] if adarms_mods is not None else None)
            suffix_out = hidden[:, -self.config.chunk_size :].to(dtype=torch.float32)
            v0 = self.action_out_proj(suffix_out)
        return noise + dt * v0

    def _denoise_steps(self, x_start, prefix_pad_masks, past_key_values, num_steps: int, start_step: int = 0) -> Tensor:
        """Forward-Euler steps ``start_step..num_steps-1`` over a filled prefix KV.

        Shared with the fused ``prefill_layer`` x ``step0_layer`` path, which resumes at step 1
        with ``x_start = x_1``. Feeds the precomputed per-step constants on the static
        (CUDA-graph) path; same values, so both paths stay bit-exact.
        """
        static = self.use_static and num_steps == self.config.num_steps
        return euler_integrate(
            lambda input_x_t, current_timestep, _time_emb=None, _mods=None, _cond=None: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
                time_emb=_time_emb,
                adarms_mods=_mods,
                adarms_cond=_cond,
            ),
            x_start,
            num_steps,
            start_step=start_step,
            time_tensors=self._time_tensors if static else None,
            time_embs=self._time_embs if static else None,
            adarms_mods=self._adarms_mods if static else None,
            adarms_conds=self._adarms_cond if static else None,
        )

    def _prefill_layer_step0_layer_overlap(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, noise, num_steps: int, device
    ) -> Tensor:
        """Prefill + step-0 fused: ``prefill_layer`` and ``step0_layer`` interleaved on two streams.

        Prefill layer ``i`` runs on the main stream, records an event, then step-0 expert layer
        ``i`` runs on the side stream behind that event (it only needs prefix KV[i]); steps
        ``1..N-1`` follow single-stream. Masks / positions / step-0 constants are built exactly
        as ``sample_actions`` and ``denoise_step`` do, so the result stays bit-identical to the
        sequential path. Not used with RTC, which rewrites the velocity per step.
        """
        vlm_with_expert = self.paligemma_with_expert
        expert = vlm_with_expert.gemma_expert.model
        num_layers = len(expert.layers)
        bsize = prefix_pad_masks.shape[0]
        dt = -1.0 / num_steps

        # ---- prefill masks / positions (identical to ``sample_actions``) ----
        prefix_len = prefix_pad_masks.shape[1]
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # ---- step-0 constants (precomputed on the static path) ----
        static = (
            self.use_static
            and num_steps == self.config.num_steps
            and self._time_tensors is not None
            and self._time_embs is not None
            and self._adarms_cond is not None
            and self._adarms_mods is not None
        )
        if static:
            time_tensor = self._time_tensors[0]  # type: ignore[index]
            time_emb = self._time_embs[0]  # type: ignore[index]
            adarms_cond = self._adarms_cond[0]  # type: ignore[index]
            # Per-layer ``dense(cond)`` modulations are built only on the static (CUDA-graph)
            # path; the eager path passes ``adarms_cond`` alone, exactly as eager
            # ``denoise_step`` does, so both sides of the overlap A/B run the same kernel tier.
            adarms_mods = self._adarms_mods[0]  # type: ignore[index]
        else:
            time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)
            time_emb = create_sinusoidal_pos_embedding(
                time_tensor,
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=device,
            ).type(dtype=torch.float32)
            adarms_cond = self.compute_adarms_cond(time_emb)
            adarms_mods = None

        # ---- step-0 suffix embedding + masks (independent of the prefill) ----
        suffix_embs, suffix_pad_masks, suffix_att_masks, _ = self.embed_suffix(
            noise, time_tensor, time_emb=time_emb, adarms_cond=adarms_cond
        )
        suffix_len = suffix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        full_att_2d_masks_4d = prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d_masks, make_att_2d_masks(suffix_pad_masks, suffix_att_masks)], dim=2)
        )
        suffix_position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # ---- interleaved layers: prefill_layer i (main) -> event -> step0_layer i (overlap) ----
        main = torch.cuda.current_stream()
        if self._overlap_stream is None:
            # Created on the first (eager) run -- warm-up / graph prime; stream and event
            # creation is not capture-safe.
            self._overlap_stream = torch.cuda.Stream()
            self._overlap_events = [torch.cuda.Event() for _ in range(num_layers)]
        overlap_stream = self._overlap_stream
        events = self._overlap_events
        assert events is not None
        join = torch.cuda.Event()

        prefix_hidden = prefix_embs
        suffix_hidden = suffix_embs
        cos_prefill = cos_step0 = None
        past_key_values: list = [None] * num_layers
        with torch.cuda.stream(overlap_stream):
            overlap_stream.wait_stream(main)  # the suffix embeds / masks were built on main
        for i in range(num_layers):
            prefix_hidden, cos_prefill, key, value = vlm_with_expert.prefill_layer(
                i, prefix_hidden, prefix_position_ids, prefix_att_2d_masks_4d, cos_prefill
            )
            past_key_values[i] = (key, value)
            events[i].record(main)
            # ``adarms_mods`` (static path only): layers see the per-layer entries; the last
            # entry is the scale/shift-only fused-tail pair.
            mods_i: tuple[Tensor, Tensor] | None = None
            if adarms_mods is not None:
                in_mod, post_mod = adarms_mods[i]
                assert post_mod is not None, "layer modulations always have a gate segment"
                mods_i = (in_mod, post_mod)
            with torch.cuda.stream(overlap_stream):
                overlap_stream.wait_event(events[i])
                suffix_hidden, cos_step0 = vlm_with_expert.step0_layer(
                    i,
                    suffix_hidden,
                    suffix_position_ids,
                    full_att_2d_masks_4d,
                    past_key_values[i],
                    adarms_cond,
                    mods_i,
                    prefix_pad_masks,
                    cos_step0,
                )
        with torch.cuda.stream(overlap_stream):
            join.record(overlap_stream)
        main.wait_event(join)

        x1 = self._finish_step0(suffix_hidden, noise, adarms_cond, adarms_mods, dt)
        return self._denoise_steps(x1, prefix_pad_masks, past_key_values, num_steps, start_step=1)

    # ------------------------------------------------------------------ #
    # ``--pad-free`` language packing (one rule for the eager path and the graphs)
    # ------------------------------------------------------------------ #
    LANG_BUCKET = 16

    def packed_lang_len(self, n_live: int) -> int:
        """Language length ``--pad-free`` packs ``n_live`` live tokens into.

        Rounded up to a ``LANG_BUCKET`` multiple (capped at ``tokenizer_max_length``) so the
        number of distinct captured graph shapes stays bounded as the task string changes.
        """
        if not self.pad_free:
            return int(self.config.tokenizer_max_length)
        bucketed = ((int(n_live) + self.LANG_BUCKET - 1) // self.LANG_BUCKET) * self.LANG_BUCKET
        return int(min(bucketed, self.config.tokenizer_max_length))

    def pack_language(self, tokens: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """Put the language tokens on the bucket grid: live prefix + zero/False tail.

        Padding rather than dropping the padded rows is load-bearing: attention over a
        different number of *masked* columns groups the softmax/GEMM reductions differently,
        so a shorter prefix would drift in bf16 from the fixed-shape captured graphs. Both
        paths must use the same length, so the eager path packs here too.

        Must not be called while capturing: the live count needs a device->host sync.
        """
        if not self.pad_free:
            return tokens, masks
        lang_len = self.packed_lang_len(int(masks.sum().item()))
        if tokens.shape[1] == lang_len and masks.shape[1] == lang_len:
            return tokens, masks
        tokens_packed = torch.zeros(tokens.shape[0], lang_len, dtype=tokens.dtype, device=tokens.device)
        masks_packed = torch.zeros(masks.shape[0], lang_len, dtype=masks.dtype, device=masks.device)
        keep = min(lang_len, int(tokens.shape[1]))
        tokens_packed[:, :keep] = tokens[:, :keep]
        masks_packed[:, :keep] = masks[:, :keep]
        return tokens_packed, masks_packed

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Full inference: prefix prefill + Euler denoising loop.

        ``inference_delay`` / ``prev_chunk_left_over`` / ``execution_horizon`` are forwarded to
        the RTC guidance hook in ``euler_integrate`` (no effect when RTC is disabled).
        """
        if num_steps is None:
            num_steps = self.config.num_steps

        bsize = tokens.shape[0]
        device = tokens.device

        # ``--pad-free`` packs the language onto the bucket grid *before* embedding so the prefix
        # length matches the captured graphs; see ``pack_language`` for why it pads, not truncates.
        if self.pad_free and not self.use_static:
            tokens, masks = self.pack_language(tokens, masks)

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = sample_noise(actions_shape, device, generator=self.noise_generator)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        rtc_enabled = self.rtc_processor is not None and self.rtc_processor.rtc_config.enabled
        if self.overlap and not rtc_enabled:
            # ``prefill_layer`` x ``step0_layer`` fusion: bit-identical to the sequential path
            # below, so the same code is correct eagerly and captured whole by ``--graph``.
            # RTC guidance rewrites the per-step velocity, so it keeps the sequential branch.
            return self._prefill_layer_step0_layer_overlap(
                prefix_embs, prefix_pad_masks, prefix_att_masks, noise, num_steps, device
            )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        assert past_key_values is not None

        return euler_integrate(
            lambda input_x_t, current_timestep, _time_emb=None, _mods=None, _cond=None: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
                time_emb=_time_emb,
                adarms_mods=_mods,
                adarms_cond=_cond,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self.rtc_processor is not None and self.rtc_processor.rtc_config.enabled,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
            time_tensors=self._time_tensors if self.use_static else None,
            time_embs=self._time_embs if self.use_static else None,
            adarms_mods=self._adarms_mods if self.use_static else None,
            adarms_conds=self._adarms_cond if self.use_static else None,
        )


# re-export for the policy / engine
def resize_with_pad_torch(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Centered-padding resize (openpi convention).

    Accepts ``[B, H, W, C]`` (channels-last) or ``[B, C, H, W]``; float32 images must be in
    [0, 1] and are clamped after resizing. Padding is centered, with the odd extra pixel on
    the bottom/right (``divmod``).
    """
    channels_last = images.shape[-1] <= 4
    if channels_last:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(images, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    resized_images = resized_images.clamp(0.0, 1.0)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = F.pad(resized_images, (pad_w0, pad_w1, pad_h0, pad_h1), value=0.0)

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images
