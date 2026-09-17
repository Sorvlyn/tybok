# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2024 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""Mixture-of-Transformers (MoT) inference: video prefill KV + action denoise.

The per-layer blocks are driven directly by this module; state keys
``mot.mixtures.<video|action>.blocks.<i>...``:
- ``prefill_video_cache``: run the video expert once, caching each layer's post-rope k/v
- ``forward_action_with_video_cache``: the action expert concatenates cached video keys
  with its own self keys

Every layer forward goes through ``FastWAMAttentionBlock.forward`` (optional args
``self_attn_mask`` / ``video_kv`` / ``context_kv`` / ``return_kv``); this module only
orchestrates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MoT(nn.Module):
    def __init__(self, mixtures: dict[str, nn.Module]):
        super().__init__()
        if set(mixtures) != {"video", "action"}:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts")
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        video = mixtures["video"]
        action = mixtures["action"]
        if len(action.blocks) != len(video.blocks):
            raise ValueError("video and action experts must have the same number of layers")
        if action.num_heads != video.num_heads or action.attn_head_dim != video.attn_head_dim:
            raise ValueError("video and action experts must share num_heads / attn_head_dim")
        self.num_layers = len(video.blocks)
        self.num_heads = video.num_heads
        self.attn_head_dim = video.attn_head_dim
        self.fp32_attention = bool(getattr(video, "fp32_attention", True))
        # video pre-fusion (--video-pre-fused): joint mask cache (config constant), set by VideoPreRunner
        self._pre_fused = None

    # ------------------------------------------------------------------ #
    # inference paths
    # ------------------------------------------------------------------ #
    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the video expert once; returns each layer's (k, v) (post-rope projections).

        Same computation as calling :meth:`prefill_video_layer` for every index, but the loop
        walks the ``blocks`` module list directly. Passing the Python ``layer_idx`` into a
        separately-traced function makes dynamo specialise on that int and compile **one graph
        per layer**, whereas iterating the modules compiles the loop once. The per-layer form
        stays for the ``prefill_video_layer`` x ``action_layer`` fusion.
        """
        payload = video_context_payload or {}
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for block in self.mixtures["video"].blocks:
            x, kv = block(
                x,
                payload.get("context"),
                video_t_mod,
                video_freqs,
                context_mask=payload.get("mask"),
                self_attn_mask=video_attention_mask,
                return_kv=True,
            )
            kv_cache.append({"k": kv[0], "v": kv[1]})
        return kv_cache

    def prefill_video_layer(
        self,
        layer_idx: int,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One video-expert prefill layer: ``(tokens_out, {"k", "v"})``.

        Per-layer form of :meth:`prefill_video_cache`, used by the fused branch
        (video layer ``i`` runs on the main stream while
        action step-0 layer ``i`` waits on this layer's KV).
        """
        video = self.mixtures["video"]
        payload = video_context_payload or {}
        block = video.blocks[layer_idx]
        x, kv = block(
            video_tokens,
            payload.get("context"),
            video_t_mod,
            video_freqs,
            context_mask=payload.get("mask"),
            self_attn_mask=video_attention_mask,
            return_kv=True,
        )
        return x, {"k": kv[0], "v": kv[1]}

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: dict | None,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        fused_runner=None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Run one step of action denoise against the cached video K/V.

        ``cross_kv``: optional per-layer precomputed cross k/v (including qk-norm); when
        not None, the per-step context kv projection is skipped. self keys = video KV ||
        this layer's keys.
        ``fused_runner``: when not None, each layer runs the fused kernels (pure bf16
        activations; requires fp8-resident weights + packed qkv + strided context k/v).
        """
        action = self.mixtures["action"]
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}.")
        if cross_kv is not None and len(cross_kv) != self.num_layers:
            raise ValueError(f"`cross_kv` must contain {self.num_layers} layers, got {len(cross_kv)}.")
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={tuple(attention_mask.shape)} vs expected_total={total_seq_len}"
            )
        # action query rows of the joint [video+action] mask
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        payload = action_context_payload or {}
        context = payload.get("context")
        context_mask = payload.get("mask")

        if fused_runner is not None:
            if cross_kv is None:
                raise ValueError(
                    "--cu-fused-adit requires the action cross-attention k/v cache "
                    "(action_context_cache); use the torch path for bit-exact A/B timing"
                )
            # The fused-kernel self-attn semantics are "video KV + all action rows
            # unmasked", consistent with build_mot_attention_mask; a mismatch is an error.
            # These assertions call .item() (D2H sync) -- skipped during CUDA graph
            # capture with validate=False.
            if validate and not bool(action_attention_mask.all().item()):
                raise ValueError(
                    "--cu-fused-adit requires an all-True action self-attention mask "
                    "(single-frame video KV), got masked rows"
                )
            if context_mask is None:
                raise ValueError("--cu-fused-adit requires a cross-attention context mask")
            if context_mask.ndim not in (2, 3):
                raise ValueError(
                    f"--cu-fused-adit: context mask must have shape [.., L], got {tuple(context_mask.shape)}"
                )
            cm_rows = context_mask.reshape(-1, context_mask.shape[-1])
            if validate and not bool((cm_rows[:1] == cm_rows).all().item()):
                raise ValueError("--cu-fused-adit requires identical cross mask rows across queries")
            return fused_runner.run_action_layers(
                action_tokens,
                action_t_mod,
                action_freqs,
                video_kv_cache,
                cross_kv,
                cm_rows[0],
            )

        x = action_tokens
        for layer_idx, block in enumerate(action.blocks):
            layer_cache = video_kv_cache[layer_idx]
            x = block(
                x,
                context,
                action_t_mod,
                action_freqs,
                context_mask=context_mask,
                self_attn_mask=action_attention_mask,
                context_kv=cross_kv[layer_idx] if cross_kv is not None else None,
                video_kv=(layer_cache["k"], layer_cache["v"]),
            )
        return x

    def prepare_action_step(
        self,
        action_t_mod: torch.Tensor,
        action_freqs: torch.Tensor,
        action_context_payload: dict | None,
        attention_mask: torch.Tensor,
        video_seq_len: int,
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        fused_runner=None,
        validate: bool = True,
    ) -> dict:
        """Per-step invariants for the per-layer action path (:meth:`action_layer`).

        Splits the setup :meth:`forward_action_with_video_cache` does once per step
        from the per-layer work. For the fused runner this also builds the kernel
        ``prep`` tuple, so it must be called on the stream the action layers will
        run on (kernels launch on the stream recorded by ``prepare``).
        """
        payload = action_context_payload or {}
        context = payload.get("context")
        context_mask = payload.get("mask")
        total_seq_len = int(attention_mask.shape[0])
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        if fused_runner is not None:
            if cross_kv is None:
                raise ValueError(
                    "--cu-fused-adit requires the action cross-attention k/v cache "
                    "(action_context_cache); use the torch path for bit-exact A/B timing"
                )
            # same contract as forward_action_with_video_cache (see there)
            if validate and not bool(action_attention_mask.all().item()):
                raise ValueError(
                    "--cu-fused-adit requires an all-True action self-attention mask "
                    "(single-frame video KV), got masked rows"
                )
            if context_mask is None:
                raise ValueError("--cu-fused-adit requires a cross-attention context mask")
            if context_mask.ndim not in (2, 3):
                raise ValueError(
                    f"--cu-fused-adit: context mask must have shape [.., L], got {tuple(context_mask.shape)}"
                )
            cm_rows = context_mask.reshape(-1, context_mask.shape[-1])
            if validate and not bool((cm_rows[:1] == cm_rows).all().item()):
                raise ValueError("--cu-fused-adit requires identical cross mask rows across queries")
            return {
                "fused": fused_runner,
                "prep": fused_runner.prepare(action_t_mod, action_freqs, cm_rows[0]),
                "cross_kv": cross_kv,
            }
        return {
            "fused": None,
            "context": context,
            "context_mask": context_mask,
            "attention_mask": action_attention_mask,
            "t_mod": action_t_mod,
            "freqs": action_freqs,
            "cross_kv": cross_kv,
        }

    def action_layer(
        self,
        layer_idx: int,
        action_tokens: torch.Tensor,
        plan: dict,
        video_kv: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One action denoise layer given this layer's video KV.

        Per-layer form of :meth:`forward_action_with_video_cache`; ``plan`` comes
        from :meth:`prepare_action_step`. Fused layers return a flat ``[M, H]``
        tensor (kernel layout), the torch path keeps the input's ``[1, M, H]``.
        """
        fused = plan["fused"]
        if fused is not None:
            cross_k, cross_v = plan["cross_kv"][layer_idx]
            return fused.step_layer_prepared(
                layer_idx, action_tokens, plan["prep"], video_kv["k"], video_kv["v"], cross_k, cross_v
            )
        block = self.mixtures["action"].blocks[layer_idx]
        return block(
            action_tokens,
            plan["context"],
            plan["t_mod"],
            plan["freqs"],
            context_mask=plan["context_mask"],
            self_attn_mask=plan["attention_mask"],
            context_kv=plan["cross_kv"][layer_idx] if plan["cross_kv"] is not None else None,
            video_kv=(video_kv["k"], video_kv["v"]),
        )

    def build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        video_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Joint [video+action] mask.

        - video rows: the video expert's own mask (first-frame-causal)
        - action rows: all action + first frame video only
        """
        pre_fused = getattr(self, "_pre_fused", None)
        if pre_fused is not None:
            # fully constant (config-determined): cached and no longer rebuilt per request (values are bitwise-identical)
            return pre_fused.mot_mask(video_seq_len, action_seq_len, video_tokens_per_frame, device)
        return self._build_mot_attention_mask_raw(
            video_seq_len, action_seq_len, video_tokens_per_frame, video_mask, device
        )

    def _build_mot_attention_mask_raw(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        video_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = video_mask
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask
