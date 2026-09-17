# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025-2026 The
# HuggingFace Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK
# in 2026. See the NOTICE file.

"""Flow-matching denoising head of the SmolVLA policy engine.

Turns an embedded VLA prefix into action chunks; see ``sample_actions``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.profiler import record_function

from ....image_utils import resize_with_pad as resize_with_pad  # re-exported (models/policy.py)
from .attention import KVCache
from .vlm_with_expert import SmolVLMWithExpertModel


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
    """Build a big_vision-style 2-D boolean attention mask."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Zero-pad the last dimension to ``new_dim`` (openpi behavior)."""
    if vector.shape[-1] == new_dim:
        return vector
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def pad_tensor(tensor: Tensor, max_len: int, pad_value=0) -> Tensor:
    """Pad a (B, L, ...) tensor along the sequence dim to ``max_len``."""
    b, d = tensor.shape[:2]
    padded_tensor = torch.full((b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :d] = tensor
    return padded_tensor


def sample_noise(shape, device, generator=None) -> Tensor:
    """Sample standard-normal float32 noise, the flow-matching ``x_1`` state.

    ``generator`` is the engine-injected ``--seed`` RNG.
    """
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device, generator=generator)


# --------------------------------------------------------------------------- #
# VLAFlowMatching
# --------------------------------------------------------------------------- #
class VLAFlowMatching(nn.Module):
    """SmolVLA action expert: flow-matching denoising over the VLM prefix.

    The engine toggles ``use_static_cache`` / ``overlap`` and the sampler.
    """

    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.vlm_with_expert = SmolVLMWithExpertModel(config, tokenizer=tokenizer)
        text_hidden = config.vlm_config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size

        self.state_proj = nn.Linear(config.max_state_dim, text_hidden, dtype=torch.float32)
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden, dtype=torch.float32)
        self.action_out_proj = nn.Linear(expert_hidden, config.max_action_dim, dtype=torch.float32)

        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden, dtype=torch.float32)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden, dtype=torch.float32)

        self.fake_image_token = None
        self.global_image_token = None
        if tokenizer is not None and tokenizer.spec is not None:
            self.fake_image_token = tokenizer.spec.fake_image_token_id
            self.global_image_token = tokenizer.spec.global_image_token_id
        self.global_image_start_token = (
            torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)
            if self.fake_image_token is not None and self.global_image_token is not None
            else None
        )

        self.add_image_special_tokens = config.add_image_special_tokens
        self.image_end_token = (
            torch.tensor([self.fake_image_token], dtype=torch.long) if self.fake_image_token is not None else None
        )
        self.prefix_length = config.prefix_length
        # ``--tl-fused-vit``: one vision-encoder forward over all cameras stacked
        # in the batch dim; drifts ~1 bf16 ulp from the serial path.
        self.vision_batch = False

        # Static-path state for CUDA-graph mode (enabled by the engine): the KV
        # buffers keep stable addresses across captures, and the precomputed
        # timestep / image-scale tensors avoid the host->device copies that are
        # illegal during capture (all of them are bit-identical to the per-forward
        # computation).
        self.use_static_cache = False
        self._kv_cache: KVCache | None = None
        self._time_embs: list[torch.Tensor] | None = None
        self._suffix_att_masks_cache: torch.Tensor | None = None
        self._suffix_att_2d_cache: torch.Tensor | None = None
        self._img_emb_scale: torch.Tensor | None = None
        # Engine-injected RNG for the denoising noise (``--seed``).
        self.noise_generator: torch.Generator | None = None

        # Denoising sampler ("euler" | "heun") and the expert prefix-KV cache.
        self.sampler: str = getattr(config, "sampler", "euler")
        self.cache_expert_prefix_kv: bool = getattr(config, "cache_expert_prefix_kv", True)
        # Per-prefix-length expert cross-attn K/V projections of the loop-invariant
        # VLM prefix KV; keyed by cache instance and ``fill_count``, so a re-fill
        # can never serve a stale projection.
        self._expert_prefix_kv: dict[int, tuple[KVCache, int, dict[int, tuple[torch.Tensor, torch.Tensor]]]] = {}
        # Shared with the VLM-with-expert forward (cross-attn layers read it).
        self.vlm_with_expert.expert_prefix_kv = self._expert_prefix_kv

        # ``overlap`` (set by the engine): prefill layer i and step-0 expert layer
        # i run on two streams joined by a per-layer event; bit-exact with the
        # sequential path, and ``--graph`` captures the whole branch per shape.
        self.overlap = False
        self._overlap_stream: torch.cuda.Stream | None = None
        self._overlap_events: list[torch.cuda.Event] | None = None

    # ------------------------------------------------------------------ #
    # prefix / suffix embedding
    # ------------------------------------------------------------------ #
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        vlm = self.vlm_with_expert

        # Batched vision encoder across cameras (drift tier); serial is bit-exact.
        if self.vision_batch and len(images) > 1:
            img_embs = vlm.embed_image(torch.cat(images, dim=0))
            img_embs = img_embs.split([img.shape[0] for img in images], dim=0)
        else:
            img_embs = [vlm.embed_image(img) for img in images]

        for img, img_mask, img_emb in zip(images, img_masks, img_embs, strict=False):
            if self.add_image_special_tokens:
                image_start_token = (
                    vlm.embed_language_tokens(self.global_image_start_token).unsqueeze(0).expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb_dim = img_emb.shape[-1]
            if self.use_static_cache:
                # Cached constant: ``torch.tensor(..., device=...)`` is a host->device
                # copy, illegal inside a CUDA graph capture.
                if self._img_emb_scale is None or self._img_emb_scale.device != img_emb.device:
                    self._img_emb_scale = torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
                img_emb = img_emb * self._img_emb_scale
            else:
                img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    vlm.embed_language_tokens(self.image_end_token).unsqueeze(0).expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        lang_emb = vlm.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # image and language inputs must not attend to state (or actions)
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        if self.use_static_cache:
            # Host copies are illegal in a capture, so build the identical
            # all-zeros / trailing-ones mask with GPU ops.
            att_masks = torch.zeros(len(att_masks), dtype=torch.bool, device=pad_masks.device)
            if states_seq_len:
                att_masks[-states_seq_len:] = True
        else:
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if self.prefix_length > 0 and seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep=None, time_emb=None):
        """Embed noisy actions and the timestep for the expert decoder.

        A precomputed ``time_emb`` (static path) takes precedence over ``timestep``.
        """
        embs = []
        pad_masks = []
        att_masks = []

        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        if time_emb is None:
            assert timestep is not None, "embed_suffix needs timestep when time_emb is not provided"
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        if self.use_static_cache:
            # Constant all-ones suffix mask, cached to avoid the host copy that
            # ``torch.tensor(list, device=...)`` would do (illegal in a capture).
            if self._suffix_att_masks_cache is None or self._suffix_att_masks_cache.device != embs.device:
                self._suffix_att_masks_cache = torch.tensor(
                    [1] * self.config.chunk_size, dtype=embs.dtype, device=embs.device
                )
            att_masks = self._suffix_att_masks_cache
        else:
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    # ------------------------------------------------------------------ #
    # denoising
    # ------------------------------------------------------------------ #
    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep=None, time_emb=None):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep, time_emb=time_emb)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        if self.use_static_cache:
            # The suffix is always fully valid, so its 2-D mask is a constant
            # all-True [1, 50, 50]; cached (bit-identical to rebuilding it).
            if self._suffix_att_2d_cache is None or self._suffix_att_2d_cache.device != suffix_pad_masks.device:
                self._suffix_att_2d_cache = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            suffix_att_2d_masks = self._suffix_att_2d_cache
        else:
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def ensure_kv_cache(self, prefix_len: int, batch_size: int = 1, device=None) -> KVCache:
        """Pre-allocate the static KV buffers for ``prefix_len`` prefix positions.

        Must happen before graph capture; a later realloc invalidates graph addresses.
        """
        vlm = self.vlm_with_expert
        if self._kv_cache is None:
            self._kv_cache = KVCache(
                num_layers=vlm.num_vlm_layers,
                num_kv_heads=vlm.num_key_value_heads,
                head_dim=vlm.head_dim,
                batch_size=batch_size,
                suffix_len=self.config.chunk_size,
            )
        if device is not None and self._kv_cache.key_buf is None:
            # force allocation with a dummy fill (overwritten by the real prefill)
            dummy = torch.empty(
                batch_size,
                prefix_len,
                vlm.num_key_value_heads,
                vlm.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            self._kv_cache.fill(0, dummy, dummy)
        return self._kv_cache

    def _time_emb_for(self, time: float, device, bsize: int, hidden: int) -> torch.Tensor:
        """Sinusoidal timestep embedding for a scalar time (fp32)."""
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        return create_sinusoidal_pos_embedding(
            time_tensor,
            hidden,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(torch.float32)

    def _precompute_time_embs(self, device, bsize: int) -> list[torch.Tensor]:
        """Timestep embeddings for the fixed ``num_steps`` schedule (static path).

        Bit-identical to per-step computation; Heun returns 2N entries (``t_i``, ``t_i + dt``).
        """
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        hidden = self.vlm_with_expert.expert_hidden_size
        time_embs = []
        for step in range(num_steps):
            t = 1.0 + step * dt
            time_embs.append(self._time_emb_for(t, device, bsize, hidden))
            if self.sampler == "heun":
                time_embs.append(self._time_emb_for(t + dt, device, bsize, hidden))
        return time_embs

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Full inference forward: prefix embedding, prefill and the denoising loop."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        return self.sample_actions_core(prefix_embs, prefix_pad_masks, prefix_att_masks, noise=noise)

    def sample_actions_core(self, prefix_embs, prefix_pad_masks, prefix_att_masks, noise=None) -> Tensor:
        """Run prefix prefill and the denoising loop on an embedded prefix.

        ``sample_actions`` = ``embed_prefix`` + this method, so CUDA-graph mode can
        capture the shape-static half (no host sync: vision position ids stay on GPU).
        """
        bsize = prefix_pad_masks.shape[0]
        device = prefix_embs.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = sample_noise(actions_shape, device, generator=self.noise_generator)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        if self.overlap and self.sampler == "euler":
            # Fused prefill x step-0 branch; bit-identical to the sequential
            # path below.
            return self._prefill_layer_step0_layer_overlap(
                prefix_embs, prefix_pad_masks, prefix_att_masks, noise, bsize, device
            )

        past_key_values, time_embs = self._ensure_denoise_ready(bsize, device)

        with record_function("vlm_with_expert.forward"):
            _, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.config.use_cache,
                fill_kv_cache=True,
            )
        if self.cache_expert_prefix_kv:
            self._compute_expert_prefix_kv(past_key_values)

        with record_function("_denoise_loop"):
            return self._denoise_loop(prefix_pad_masks, past_key_values, noise, time_embs, bsize, device)

    # prefill_layer x step0_layer fusion (``overlap``)
    # --------------------------------------------------------------------------- #
    def _prefill_layer_step0_layer_overlap(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, noise, bsize: int, device
    ) -> Tensor:
        """Interleave prefill layer ``i`` and step-0 expert layer ``i`` on two streams.

        The engine sets ``overlap`` from ``--no-overlap``. Step-0 layer ``i`` reads prefix
        KV[i], so it waits on an event recorded after prefill layer ``i``; steps 1..N-1
        stay single-stream. Bit-exact with the sequential path. Euler only: Heun keeps
        the sequential path, and ``--graph`` captures the fork/join as graph edges.
        """
        vlm = self.vlm_with_expert
        num_layers = vlm.num_vlm_layers
        dt = -1.0 / self.config.num_steps

        past_key_values, time_embs = self._ensure_denoise_ready(bsize, device)
        if past_key_values is None:
            # Fresh cache, like the eager non-static path.
            past_key_values = KVCache(
                num_layers=vlm.num_vlm_layers, num_kv_heads=vlm.num_key_value_heads, head_dim=vlm.head_dim
            )
        if time_embs is None:
            t0_emb = self._time_emb_for(1.0, device, bsize, vlm.expert_hidden_size)
        else:
            t0_emb = time_embs[0]

        # ---- step-0 suffix embedding + masks (independent of the prefill) ----
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(noise, time_emb=t0_emb)
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suf_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # ---- prefill masks / positions ----
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        main = torch.cuda.current_stream()
        if self._overlap_stream is None:
            self._overlap_stream = torch.cuda.Stream()
            self._overlap_events = [torch.cuda.Event() for _ in range(num_layers)]
        overlap_stream = self._overlap_stream
        events = self._overlap_events
        assert events is not None
        join = torch.cuda.Event()

        prefix_hidden = prefix_embs
        suffix_hidden = suffix_embs
        with torch.cuda.stream(overlap_stream):
            overlap_stream.wait_stream(main)  # suffix embeds / masks were built on main
        for i in range(num_layers):
            # main: LLM prefill layer i (writes prefix KV[i])
            prefix_hidden, past_key_values = vlm.prefill_layer(
                i, prefix_hidden, prefix_position_ids, att_2d_masks, bsize, past_key_values
            )
            events[i].record(main)
            # step-0 expert layer i on the overlap stream, gated on prefix KV[i]
            with torch.cuda.stream(overlap_stream):
                overlap_stream.wait_event(events[i])
                suffix_hidden, past_key_values = vlm.step0_layer(
                    i, suffix_hidden, suf_position_ids, full_att_2d_masks, bsize, past_key_values
                )
        with torch.cuda.stream(overlap_stream):
            join.record(overlap_stream)
        main.wait_event(join)

        # ---- finish step 0: final norm + velocity projection + Euler update ----
        # (``step0_layer`` returns the pre-norm hidden state.)
        suffix_hidden = self.vlm_with_expert.lm_expert.norm(suffix_hidden)
        suffix_out = suffix_hidden[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v0 = self.action_out_proj(suffix_out)
        x1 = noise + dt * v0
        if self.cache_expert_prefix_kv:
            self._compute_expert_prefix_kv(past_key_values)
        return self._denoise_loop(prefix_pad_masks, past_key_values, x1, time_embs, bsize, device, start_step=1)

    def _ensure_denoise_ready(self, bsize: int, device):
        """Static-path cache plumbing shared by the eager and split-graph paths."""
        if self.use_static_cache:
            if self._kv_cache is None:
                self._kv_cache = KVCache(
                    num_layers=self.vlm_with_expert.num_vlm_layers,
                    num_kv_heads=self.vlm_with_expert.num_key_value_heads,
                    head_dim=self.vlm_with_expert.head_dim,
                    batch_size=bsize,
                    suffix_len=self.config.chunk_size,
                )
            past_key_values = self._kv_cache
            if self._time_embs is None:
                self._time_embs = self._precompute_time_embs(device, bsize)
            time_embs = self._time_embs
        else:
            past_key_values = None
            time_embs = None
        return past_key_values, time_embs

    def _compute_expert_prefix_kv(self, past_key_values: KVCache) -> None:
        """Project the loop-invariant prefix K/V once per inference.

        The prefix KV is fixed after prefill, so the fp32 projections match recomputing
        them every step. Entries are keyed by prefix length and bound to the cache
        instance and its ``fill_count``, so a re-fill cannot serve a stale entry.
        """
        prefix_len = past_key_values.prefix_len
        layer_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        vlm = self.vlm_with_expert
        for layer_idx in range(vlm.num_vlm_layers):
            if "cross" not in vlm.attention_mode or layer_idx % vlm.self_attn_every_n_layers == 0:
                continue  # self-attention layer: no expert prefix projection
            layer = vlm.lm_expert.layers[layer_idx]
            key_states, value_states = (
                past_key_values[layer_idx]["key_states"],
                past_key_values[layer_idx]["value_states"],
            )
            _key_states = key_states.to(dtype=layer.self_attn.k_proj.weight.dtype).view(*key_states.shape[:2], -1)
            expert_key = layer.self_attn.k_proj(_key_states).view(*_key_states.shape[:-1], -1, layer.self_attn.head_dim)
            _value_states = value_states.to(dtype=layer.self_attn.v_proj.weight.dtype).view(*value_states.shape[:2], -1)
            expert_value = layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, layer.self_attn.head_dim
            )
            layer_cache[layer_idx] = (expert_key, expert_value)
        self._expert_prefix_kv[prefix_len] = (past_key_values, past_key_values.fill_count, layer_cache)

    def _denoise_loop(
        self, prefix_pad_masks, past_key_values, noise, time_embs, bsize: int, device, start_step: int = 0
    ) -> Tensor:
        """Run Euler or Heun denoising over a prefilled prefix.

        Euler: ``x <- x + dt * v(x, t)`` with ``dt = -1/num_steps``; Heun adds a
        corrector eval at ``t + dt``. ``start_step`` resumes at step 1 for the
        fused branch, where ``noise`` is then the ``x_1`` state.
        """
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        if self.sampler == "heun":
            assert start_step == 0, "Heun keeps the sequential path (no overlap)"
            for step in range(num_steps):
                if time_embs is None:
                    t = 1.0 + step * dt
                    time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
                    v0 = self.denoise_step(
                        x_t=x_t,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        timestep=time_tensor,
                    )
                    time1_tensor = torch.tensor(t + dt, dtype=torch.float32, device=device).expand(bsize)
                    v1 = self.denoise_step(
                        x_t=x_t + dt * v0,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        timestep=time1_tensor,
                    )
                else:
                    v0 = self.denoise_step(
                        x_t=x_t,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        time_emb=time_embs[2 * step],
                    )
                    v1 = self.denoise_step(
                        x_t=x_t + dt * v0,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        time_emb=time_embs[2 * step + 1],
                    )
                x_t = x_t + (dt * 0.5) * (v0 + v1)
        else:
            for step in range(start_step, num_steps):
                if time_embs is None:
                    t = 1.0 + step * dt
                    time_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
                    v_t = self.denoise_step(
                        x_t=x_t,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        timestep=time_tensor,
                    )
                else:
                    v_t = self.denoise_step(
                        x_t=x_t,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        time_emb=time_embs[step],
                    )
                x_t = x_t + dt * v_t
        return x_t
