# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2024 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""FastWAM world model -- action inference.

Combines the MoT experts (video + action DiT), the proprio encoder, and the frozen
Wan components (VAE encoder, UMT5 text encoder, tokenizer).  Inference pipeline:

1. Wan VAE-encode the current camera frame -> first-frame latents
2. tokenize + UMT5-encode the task prompt; append the normalized proprio as an extra context token
3. run the *video expert* once over the first-frame latents at timestep 0, caching per-layer post-rope k/v
4. denoise ``action_horizon`` actions over ``num_inference_steps`` Euler steps (Wan sigma-shift flow matching), reusing the video KV cache

Only ``mot`` + ``proprio_encoder`` are registered submodules (their keys live in the
policy ``model.safetensors``); the frozen VAE / text encoder are plain attributes.
State-dict layout: ``model.mot.mixtures.<video|action>...`` + ``model.proprio_encoder``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import FastWAMConfig
from .action_dit import ActionDiT
from .mot import MoT
from .video_dit import WanVideoDiT

log = logging.getLogger("tybok.fastwam")


def get_sampling_sigmas(sampling_steps: int, shift: float):
    """Wan-compatible sigma schedule: uniform [1, 0] grid warped by the shift."""
    import numpy as np

    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


class FastWAMScheduler:
    """Continuous flow-matching inference schedule (Wan ``infer_shift``)."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift = self.shift if shift_override is None else float(shift_override)
        sigma_steps = torch.as_tensor(
            get_sampling_sigmas(num_inference_steps, shift), device=device, dtype=torch.float32
        )
        timesteps = sigma_steps * float(self.num_train_timesteps)
        sigma_next = torch.cat([sigma_steps[1:], sigma_steps.new_zeros(1)])
        deltas = sigma_next - sigma_steps
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta


class FastWAM(nn.Module):
    """FastWAM action-inference model (batch 1)."""

    def __init__(
        self,
        config: FastWAMConfig,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        action_context_cache: bool = True,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.action_context_cache = bool(action_context_cache)

        # ``prefill_video_layer`` x ``action_layer`` fusion (set by the engine; CUDA only):
        # video prefill layer i runs on the main stream while step-0 action layer i runs on a
        # side stream, gated on prefill layer i's video KV via an event. The same flag and the
        # same fork/join shape as smolvla/pi05 (``model.overlap``); bit-exact with the
        # sequential path (same ops, same data dependencies).
        self.overlap = False
        self._overlap_stream: torch.cuda.Stream | None = None
        self._overlap_events: list[torch.cuda.Event] | None = None

        video_expert = WanVideoDiT(config.video)
        action_expert = ActionDiT(config.action)
        self.mot = MoT({"video": video_expert, "action": action_expert})
        # unregistered aliases (like the reference; mot is the single owner)
        object.__setattr__(self, "video_expert", video_expert)
        object.__setattr__(self, "action_expert", action_expert)
        object.__setattr__(self, "dit", self.mot)

        text_dim = int(config.video.text_dim)
        self.text_dim = text_dim
        if config.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(int(config.proprio_dim), text_dim)
        else:
            self.proprio_encoder = None
        self.proprio_dim = config.proprio_dim

        # frozen side components (never registered -> excluded from state_dict)
        object.__setattr__(self, "vae", vae)
        object.__setattr__(self, "text_encoder", text_encoder)
        self.tokenizer = tokenizer
        # Prompt-embedding memo: ``prompt -> (context, context_mask)`` for the **one** prompt
        # in use. A deployment repeats the same task string on every request, so a single
        # entry hits every time after the first yet keeps memory bounded to one embedding;
        # the warm-up call seeds it and a different task replaces it. The cached tensors are
        # handed out as-is (no per-request clone, so a cache hit also skips the HtoD copy):
        # callers must treat them as read-only -- ``_append_proprio_to_context`` appends with
        # ``torch.cat`` and the DiT only reads the context, so nothing mutates them in place.
        self._prompt_cache: tuple[str, torch.Tensor, torch.Tensor] | None = None
        # ``False`` (``--no-prompt-cache``) encodes the task on every request instead of
        # reusing ``_prompt_cache``. Still bit-identical -- the memo stores exactly what a
        # fresh encode returns -- and exists so profiling / benchmarking can measure the real
        # per-request cost including the UMT5 forward. The engine sets it from its own flag.
        self.cache_prompt = True

        self.infer_action_scheduler = FastWAMScheduler(
            num_train_timesteps=config.action_scheduler.num_train_timesteps,
            shift=config.action_scheduler.infer_shift,
        )

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """UMT5-encode the padded prompt ids, zero every pad-row embedding, then
        return an all-ones mask (cross-attention runs over a fixed-length context).

        The text encoder may live on a different device than the DiT; ids are moved
        to the encoder's device and the context back to the model device.

        The result is memoized for the current prompt (``self._prompt_cache``): the UMT5
        forward is the whole per-request cost when the text encoder stays on CPU (~9s on
        the reference checkpoint), and a deployment asks for the same prompt every time.
        ``self.cache_prompt=False`` (``--no-prompt-cache``) turns the memo off, so every
        request pays a real UMT5 forward (for profiling / benchmarking). The returned tensors
        are the cache's own when the memo is on, so callers must treat them as read-only.
        """
        cached = self._prompt_cache if self.cache_prompt else None
        if cached is not None and cached[0] == prompt:
            return cached[1], cached[2]
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("prompt encoding requires the text encoder/tokenizer")
        # ``te_device`` is where the *ids* have to go: the embedding table may be parked on CPU
        # on purpose (``--text-emb-cpu``), so it is not necessarily where the forward runs -- the
        # log below therefore reads ``text_encoder_device`` (set by the engine) instead.
        te_device = next(self.text_encoder.parameters()).device
        ids, mask = self.tokenizer.encode_prompt(prompt)
        ids = ids.to(te_device)
        mask = mask.to(te_device)
        prompt_emb = self.text_encoder(ids, mask.to(dtype=torch.long))
        seq_lens = mask.long().sum(dim=1)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, int(v) :] = 0
        prompt_emb = prompt_emb.to(device=self.device)
        mask = torch.ones_like(mask).to(device=self.device)
        if self.cache_prompt:
            self._prompt_cache = (prompt, prompt_emb, mask)
            tail = "cached for follow-up requests"
        else:
            tail = "prompt cache disabled (--no-prompt-cache)"
        log.info(
            f"prompt embedding encoded (task={prompt!r}, tokens={int(prompt_emb.shape[1])}, "
            f"text encoder on {getattr(self, 'text_encoder_device', te_device)}); {tail}"
        )
        return prompt_emb, mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` must be 2D [B, {self.proprio_dim}], got {tuple(proprio.shape)}")
        proprio_token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)).to(
            dtype=context.dtype
        )
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _prepare_infer_context(
        self, prompt: str | None, proprio: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            raise ValueError("fastwam deploy requires a task prompt (the model is not unconditional here)")
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        return context, context_mask

    @torch.no_grad()
    def _encode_input_image_latents(self, input_image: torch.Tensor) -> torch.Tensor:
        """Image [0,1] -> Wan VAE standardized first-frame latent [1,48,1,h,w]."""
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        image = (input_image * 2.0 - 1.0).to(device=self.device, dtype=self.torch_dtype)
        z = self.vae.encode_frame(image.unsqueeze(2))
        return z.to(device=self.device)

    def _make_action_latents(
        self, action_horizon: int, seed: int | None, rand_device: str, noise: torch.Tensor | None
    ) -> torch.Tensor:
        if noise is not None:
            noise = noise.to(device=self.device, dtype=self.torch_dtype)
            if tuple(noise.shape) != (1, action_horizon, self.action_expert.action_dim):
                raise ValueError(
                    "`noise` must have shape [1, action_horizon, action_dim] = "
                    f"[1, {action_horizon}, {self.action_expert.action_dim}], got {tuple(noise.shape)}"
                )
            return noise.contiguous()
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        return torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def infer_action(
        self,
        input_image: torch.Tensor,
        action_horizon: int,
        prompt: str | None = None,
        proprio: torch.Tensor | None = None,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if str(self.video_expert.video_attention_mask_mode) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")

        input_image, height, width = self._normalize_input_image(input_image)
        if proprio is None:
            proprio_t = None
        else:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio_t = proprio.to(device=self.device, dtype=self.torch_dtype)

        latents_action = self._make_action_latents(action_horizon, seed, rand_device, noise)

        context, context_mask = self._prepare_infer_context(prompt, proprio_t)

        # CUDA graph covers the VAE frame encode + the DiT core (video prefill + action denoise
        # loop, fixed shape, GPU-only) and now takes the **normalized image** as its input; the
        # tokenizer / text encoder / VAE ``.encode_frame`` host-side entry points stay outside.
        graph = getattr(self, "_graph_runner", None)
        if graph is not None:
            latents_action = graph.run(
                latents_action,
                input_image,
                context,
                context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
            )
        else:
            first_frame_latents = self._encode_input_image_latents(input_image)
            latents_action = self._denoise_core(
                latents_action,
                first_frame_latents,
                context,
                context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                validate=True,
            )

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    def _denoise_core(
        self,
        latents_action: torch.Tensor,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        schedule: tuple[torch.Tensor, torch.Tensor] | None = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """video-expert prefill + action denoise loop, returns the final ``latents_action``
        (GPU).

        **CUDA-graph-capturable section**: fixed shapes, no CPU sync. ``validate=False``
        skips the defensive assertions that call ``.item()`` (D2H sync) -- they run once in
        the eager round before graph capture. The text side (tokenizer + text encoder, the
        embedding layer defaults to CPU) is outside this section. The VAE frame encode that
        produces ``first_frame_latents`` is outside the *eager* call here, but the graph
        runner captures it together with this method (see ``graph_runner``), which is why
        ``infer_action`` hands the runner the normalized image instead of the latents.

        When ``self.overlap`` is set, the first denoise step is fused into the video
        prefill (see :meth:`_prefill_video_layer_action_layer_overlap`); steps 1..N-1 are unchanged.
        """
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # video-expert prefill (timestep 0), per-layer KV cache
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        attention_mask = self.mot.build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            video_mask=video_mask,
            device=video_pre["tokens"].device,
        )

        fused_runner = getattr(self, "_action_fused_runner", None)
        # context-side work invariant across denoise steps (text embedding + each layer's cross kv)
        action_ctx: dict | None = None
        if self.action_context_cache:
            action_ctx = self.action_expert.precompute_action_context(context, context_mask)
            if fused_runner is not None:
                # same as above: the structure is the same across layers, checking layer 0 suffices.
                _k0, _v0 = action_ctx["kv"][0]
                if (
                    _k0.dtype != torch.bfloat16
                    or not _k0.is_contiguous()
                    or _v0.dtype != torch.bfloat16
                    or not _v0.is_contiguous()
                ):
                    action_ctx["kv"] = [
                        (k.to(dtype=torch.bfloat16).contiguous(), v.to(dtype=torch.bfloat16).contiguous())
                        for k, v in action_ctx["kv"]
                    ]

        # action denoising schedule
        if schedule is None:
            infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        else:
            # graph-capture path: schedule is precomputed outside the graph (avoids pageable H2D during capture)
            infer_timesteps, infer_deltas = schedule

        if self.overlap:
            # video prefill layer i (main stream) x step-0 action layer i (side stream):
            # step 0 hides inside the prefill window.
            video_kv_cache, latents_action = self._prefill_video_layer_action_layer_overlap(
                latents_action=latents_action,
                video_pre=video_pre,
                video_seq_len=video_seq_len,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                action_ctx=action_ctx,
                fused_runner=fused_runner,
                timestep0=infer_timesteps[0],
                delta0=infer_deltas[0],
                validate=validate,
            )
            first_step = 1
        else:
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
            first_step = 0

        # the fused kernels require contiguous bf16: flash reads contiguous [L_video, H3]
        # with row stride H3, while prefill's v is a strided slice of the concatenated qkv
        # buffer (row stride 3*H3) -- a layout contract violation (not a race) that needs a
        # defensive conversion; fp32 k is also converted to bf16.
        if fused_runner is not None:
            # layout is the same across layers: checking layer 0 suffices; if already
            # contiguous, skip the whole block (no sync, graph-safe).
            _c0 = video_kv_cache[0]
            if (
                _c0["k"].dtype != torch.bfloat16
                or not _c0["k"].is_contiguous()
                or _c0["v"].dtype != torch.bfloat16
                or not _c0["v"].is_contiguous()
            ):
                video_kv_cache = [
                    {
                        "k": c["k"].to(dtype=torch.bfloat16).contiguous(),
                        "v": c["v"].to(dtype=torch.bfloat16).contiguous(),
                    }
                    for c in video_kv_cache
                ]

        # action denoising (step 0 already done above when overlap is on)
        for step_idx in range(first_step, len(infer_timesteps)):
            step_t = infer_timesteps[step_idx]
            step_delta = infer_deltas[step_idx]
            timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self.denoise_step(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_ctx=action_ctx,
                validate=validate,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)

        return latents_action

    def _prefill_video_layer_action_layer_overlap(
        self,
        latents_action: torch.Tensor,
        video_pre: dict,
        video_seq_len: int,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_ctx: dict | None,
        fused_runner,
        timestep0: torch.Tensor,
        delta0: torch.Tensor,
        validate: bool = True,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """``prefill_video_layer`` with denoise step 0's ``action_layer`` fused in (two streams).

        Video prefill layer ``i`` writes its video KV on the main stream; step-0 action
        layer ``i`` only reads that layer's KV (self-attention concatenates the video KV
        with its own keys), so it is gated on an event recorded right after prefill layer
        ``i`` and runs on a side stream inside the prefill window. Returns the per-layer
        video KV cache (for steps ``1..N-1``) and the latent after step 0.

        This is the eager form of the same fusion smolvla/pi05 implement (their
        ``_prefill_layer_step0_layer_overlap``), and the same code is what
        `FastWAMGraphRunner(overlap=True)` captures (stream capture turns the fork/join
        events into graph edges, so the interleave survives as a single multi-stream graph).
        On GPUs with saturated SMs it is a latency win only when the kernels can co-run
        (the cooperative `--cu-fused-*` ones cannot). Same ops / same data dependencies as
        the sequential path, so the result is bit-exact.
        """
        mot = self.mot
        action = self.action_expert

        # step-0 action pre_dit depends only on the noise/timestep/context, not on the
        # video prefill, so it is ready before the first layer.
        timestep_action = timestep0.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
        action_pre = action.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            context_emb=action_ctx["emb"] if action_ctx else None,
        )

        video_payload = {"context": video_pre["context"], "mask": video_pre["context_mask"]}
        video_attn_mask = attention_mask[:video_seq_len, :video_seq_len]

        main = torch.cuda.current_stream()
        if self._overlap_stream is None:
            self._overlap_stream = torch.cuda.Stream()
        if self._overlap_events is None or len(self._overlap_events) != mot.num_layers:
            self._overlap_events = [torch.cuda.Event() for _ in range(mot.num_layers)]
        overlap_stream = self._overlap_stream
        events = self._overlap_events

        # The plan is built on the overlap stream: the fused runner records the stream its
        # kernels launch on in ``prepare``, and the step-0 layers run there.
        with torch.cuda.stream(overlap_stream):
            overlap_stream.wait_stream(main)  # pre_dit output / context / mask are read there
            plan = mot.prepare_action_step(
                action_t_mod=action_pre["t_mod"],
                action_freqs=action_pre["freqs"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                cross_kv=action_ctx["kv"] if action_ctx else None,
                fused_runner=fused_runner,
                validate=validate,
            )

        video_kv_cache: list[dict[str, torch.Tensor]] = []
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        for layer_idx in range(mot.num_layers):
            # main: prefill layer i (writes video KV[i])
            video_tokens, kv_i = mot.prefill_video_layer(
                layer_idx,
                video_tokens,
                video_pre["freqs"],
                video_pre["t_mod"],
                video_payload,
                video_attn_mask,
            )
            video_kv_cache.append(kv_i)
            events[layer_idx].record(main)
            # overlap stream: step-0 action layer i, gated on video KV[i]
            with torch.cuda.stream(overlap_stream):
                overlap_stream.wait_event(events[layer_idx])
                action_tokens = mot.action_layer(layer_idx, action_tokens, plan, kv_i)

        main.wait_stream(overlap_stream)  # step-0 expert chain done -> the head may read it
        # fused layers return a flat [M, H]; restore the batched [1, M, H] the head expects
        if action_tokens.ndim == 2:
            action_tokens = action_tokens.view(1, action_tokens.shape[0], action_tokens.shape[1])
        pred_action = action.post_dit(action_tokens, action_pre)
        latents_action = self.infer_action_scheduler.step(pred_action, delta0, latents_action)
        return video_kv_cache, latents_action

    def denoise_step(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_ctx: dict | None = None,
        validate: bool = True,
    ) -> torch.Tensor:
        context_emb = action_ctx["emb"] if action_ctx else None
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            context_emb=context_emb,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            cross_kv=action_ctx["kv"] if action_ctx else None,
            fused_runner=getattr(self, "_action_fused_runner", None),
            validate=validate,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @staticmethod
    def _normalize_input_image(input_image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        return input_image, height, width


def build_fastwam(
    config: FastWAMConfig,
    vae,
    text_encoder,
    tokenizer,
    device: str | torch.device,
    torch_dtype: torch.dtype,
) -> FastWAM:
    return FastWAM(
        config=config,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
    )
