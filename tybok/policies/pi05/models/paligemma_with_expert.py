# Derived from LeRobot <https://github.com/huggingface/lerobot> and transformers
# <https://github.com/huggingface/transformers>. Copyright 2025 Physical Intelligence and The
# HuggingFace Inc. team.; Copyright 2024 Google Inc. HuggingFace Inc. team. Licensed under the
# Apache License, Version 2.0; modified for TyBoK in 2026. See the NOTICE file.

"""PaliGemma backbone + action expert for pi0.5.

The module tree matches the checkpoint parameter paths (``paligemma.model.vision_tower.vision_model``,
``paligemma.model.language_model``, ``gemma_expert.model``, ...) so the generic ``safetensors``
loader can match keys by name.

``embed_image`` -> projected per-patch features ``(B, 256, 2048)`` f32.
``embed_language_tokens`` -> the VLM embedding table (tied lm_head).
``forward`` runs either the VLM decoder over the prefix (``inputs_embeds[0]``,
``use_cache=True`` -> per-layer prefix KV) or the expert decoder over the suffix
with that prefix KV and the AdaRMS conditioning.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import PI05Config
from .gemma import GemmaDecoder
from .vision import SiglipVisionTransformer

# PaliGemma vocabulary size (lm_head / embed table).
PALIGEMMA_VOCAB_SIZE = 257152


class PaliGemmaMultiModalProjector(nn.Module):
    """Single linear (1152 -> projection_dim), as in transformers 5.5.4 (f32)."""

    def __init__(self, hidden_size: int, projection_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, projection_dim, bias=True, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PaliGemmaModel(nn.Module):
    """``paligemma.model``: vision tower + projector + language decoder."""

    def __init__(self, config: PI05Config):
        super().__init__()
        vlm = config.paligemma
        # SigLIP-SO400M vision tower (f32): 27 layers, patch 14, image 224.
        self.vision_tower = nn.Module()
        self.vision_tower.vision_model = SiglipVisionTransformer(
            hidden_size=1152,
            num_heads=16,
            num_layers=27,
            intermediate_size=4304,
            image_size=config.image_resolution[0],
            patch_size=14,
        )
        self.multi_modal_projector = PaliGemmaMultiModalProjector(1152, vlm.width)
        # Fused post_layernorm + projector Triton path.
        self._fused_projector = False
        self._proj_w: torch.Tensor | None = None
        self._proj_b: torch.Tensor | None = None
        self.language_model = GemmaDecoder(
            vlm.width,
            vlm.depth,
            vlm.mlp_dim,
            vlm.num_heads,
            vlm.num_kv_heads,
            vlm.head_dim,
            use_adarms=False,
            vocab_size=PALIGEMMA_VOCAB_SIZE,
        )

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> projected last hidden state (B, 256, projection_dim) f32."""
        if self._fused_projector:
            from .triton_vision import triton_vision_projector

            tower = self.vision_tower.vision_model
            hidden = tower.encode(pixel_values)  # [B, 256, 1152] (pre-post_layernorm)
            B, L, C = hidden.shape
            ln = tower.post_layernorm
            return triton_vision_projector(
                hidden.reshape(-1, C), ln.weight, ln.bias, self._proj_w, self._proj_b, ln.eps
            ).reshape(B, L, -1)
        selected_image_feature = self.vision_tower.vision_model(pixel_values)
        return self.multi_modal_projector(selected_image_feature)

    def set_fused_projector(self) -> None:
        """Fuse ``post_layernorm + multi_modal_projector`` into one bf16 Triton GEMM.

        The projector runs bf16 but writes fp32, so the VLM prefix is unchanged.
        """
        lin = self.multi_modal_projector.linear
        self._proj_w = lin.weight.data.to(torch.bfloat16).contiguous()
        self._proj_b = lin.bias.data.to(torch.bfloat16).contiguous()
        self._fused_projector = True


class PaliGemmaForConditionalGeneration(nn.Module):
    """``paligemma`` wrapper so the checkpoint's ``paligemma.model.*`` paths resolve."""

    def __init__(self, config: PI05Config):
        super().__init__()
        self.model = PaliGemmaModel(config)


class GemmaCausalLM(nn.Module):
    """``gemma_expert`` wrapper (the checkpoint stores the expert under ``gemma_expert.model``)."""

    def __init__(self, config: PI05Config):
        super().__init__()
        expert = config.expert
        self.model = GemmaDecoder(
            expert.width,
            expert.depth,
            expert.mlp_dim,
            expert.num_heads,
            expert.num_kv_heads,
            expert.head_dim,
            use_adarms=True,
            vocab_size=None,
        )


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, config: PI05Config):
        super().__init__()
        self.paligemma = PaliGemmaForConditionalGeneration(config)
        self.gemma_expert = GemmaCausalLM(config)
        # Cached embedding scale (sqrt(hidden_size) as bf16); building it per call is a
        # host->device copy inside CUDA-graph capture (cached value is bit-identical).
        self._embed_scale: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """image (B, 3, H, W) f32 in [-1, 1] -> (B, 256, vlm.width) f32."""
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        features = self.paligemma.model.get_image_features(image)
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        assert self.paligemma.model.language_model.embed_tokens is not None
        emb = self.paligemma.model.language_model.embed_tokens(tokens)
        # Gemma scales embeddings by sqrt(hidden_size); it goes through the bf16
        # conversion, so it is 45.25 (bf16), not the exact sqrt(2048).
        scale = self._embed_scale
        if scale is None or scale.dtype != emb.dtype or scale.device != emb.device:
            scale = torch.tensor(self.paligemma.model.language_model.width**0.5, dtype=emb.dtype, device=emb.device)
            self._embed_scale = scale
        if scale.is_inference() and not torch.is_inference_mode_enabled():
            # The cache may be an inference tensor (created inside inference_mode warmup /
            # capture); materialize a normal tensor when used outside.
            scale = torch.tensor(scale.cpu().item(), dtype=scale.dtype, device=scale.device)
            self._embed_scale = scale
        return emb * scale

    # ------------------------------------------------------------------ #
    # per-layer primitives for the ``prefill_layer`` x ``step0_layer`` fusion
    # (``PI05FlowMatching.overlap``): one call == one layer, so the LLM prefill of
    # layer ``i`` can run on the main stream while the step-0 expert layer ``i`` runs on a side
    # stream, gated on the prefix KV that prefill layer ``i`` just wrote. Both must be pure
    # functions of their inputs (no state beyond what they return).
    # ------------------------------------------------------------------ #
    def prefill_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """One LLM prefill layer -> ``(hidden, cos_sin, key_states, value_states)``.

        Layer 0 casts the prefix embeddings to the layer dtype and builds the shared RoPE
        ``cos/sin`` -- exactly as ``GemmaDecoder.forward`` does once for the whole stack, so
        the per-layer chain is bit-identical to the sequential prefill.
        """
        vlm = self.paligemma.model.language_model
        if layer_idx == 0:
            if vlm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype=torch.bfloat16)
            cos_sin = vlm.rotary_emb(hidden_states, position_ids)
        assert cos_sin is not None
        hidden_states, key_states, value_states = vlm.layers[layer_idx](
            hidden_states, attention_mask, position_ids, None, cos_sin, None, None, None
        )
        return hidden_states, cos_sin, key_states, value_states

    def step0_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor],
        adarms_cond: torch.Tensor,
        adarms_mods: tuple[torch.Tensor, torch.Tensor] | None,
        prefix_pad_masks: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One expert layer of the first denoising step -> ``(hidden, cos_sin)``.

        Cross-attends ``prefix_kv`` (the prefix KV of this same layer index, written by
        ``prefill_layer``) and applies the step-0 AdaRMS conditioning. The suffix's shared RoPE
        ``cos/sin`` is built by layer 0.

        ``adarms_cond`` / ``adarms_mods`` are forwarded as-is: the static (CUDA-graph) path passes
        the precomputed per-layer modulations (the fused Triton expert branch), the eager path
        passes the conditioning vector alone -- the same split the sequential ``denoise_step``
        makes, so both sides of the overlap comparison run the same kernels.
        """
        expert = self.gemma_expert.model
        if layer_idx == 0:
            if expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype=torch.bfloat16)
            cos_sin = expert.rotary_emb(hidden_states, position_ids)
        assert cos_sin is not None
        hidden_states, _, _ = expert.layers[layer_idx](
            hidden_states,
            attention_mask,
            position_ids,
            prefix_kv,
            cos_sin,
            adarms_cond,
            adarms_mods,
            prefix_pad_masks,
        )
        return hidden_states, cos_sin

    # ------------------------------------------------------------------ #
    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool = False,
        adarms_cond: list[torch.Tensor | None] | None = None,
        adarms_mods: list[list[tuple[torch.Tensor, torch.Tensor]] | None] | None = None,
        triton_prefix_pad: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Returns ([prefix_output, suffix_output], past_key_values)."""
        if adarms_cond is None:
            adarms_cond = [None, None]
        if adarms_mods is None:
            adarms_mods = [None, None]
        if triton_prefix_pad is None:
            triton_prefix_pad = [None, None]
        if inputs_embeds[1] is None:
            prefix_output, kv = self.paligemma.model.language_model(
                inputs_embeds[0],
                attention_mask,
                position_ids,
                prefix_kv=None,
                adarms_cond=adarms_cond[0],
                use_cache=use_cache,
            )
            return [prefix_output, None], kv
        else:
            suffix_output, _ = self.gemma_expert.model(
                inputs_embeds[1],
                attention_mask,
                position_ids,
                prefix_kv=past_key_values,
                adarms_cond=adarms_cond[1],
                adarms_mods=adarms_mods[1],
                triton_prefix_pad=triton_prefix_pad[1],
                use_cache=False,
            )
            return [None, suffix_output], None
