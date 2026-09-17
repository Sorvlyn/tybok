# Derived from LeRobot <https://github.com/huggingface/lerobot>. Copyright 2025 The HuggingFace
# Inc. team. Licensed under the Apache License, Version 2.0; modified for TyBoK in 2026. See
# the NOTICE file.

"""SmolVLM backbone + action expert with the shared-attention forward.

Attention runs here, not in the ``transformers`` Llama layers; the engine's
prefix prefill / denoising steps and the ``--compile`` regions run through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .attention import EagerAttention, KVCache
from .expert import ExpertModel
from .rope import apply_rope
from .text import TextModel
from .vision import Connector, VisionTransformer

if TYPE_CHECKING:
    from ..config import SmolVLAConfig


class _VLM(nn.Module):
    """Holds the vision encoder, connector and text decoder under ``vlm.model``."""

    def __init__(self, vlm_config, text_config, vision_config):
        super().__init__()
        self.config = vlm_config
        self.model = nn.Module()
        self.model.vision_model = VisionTransformer(vision_config)
        self.model.connector = Connector(vlm_config)
        self.model.text_model = TextModel(text_config)


class SmolVLMWithExpertModel(nn.Module):
    def __init__(self, config: "SmolVLAConfig", tokenizer=None):
        super().__init__()
        assert config.vlm_config is not None, "vlm_config is required (set VLM dir in config.json)"
        text_config = config.vlm_config.text_config
        vision_config = config.vlm_config.vision_config

        self.num_vlm_layers = config.num_vlm_layers
        self.num_expert_layers = config.n_expert_layers
        self.self_attn_every_n_layers = config.self_attn_every_n_layers
        self.attention_mode = config.attention_mode
        self.expert_hidden_size = config.expert_hidden_size

        # keep only the first num_vlm_layers decoder layers
        text_config_full = config.vlm_config.text_config
        self.vlm = _VLM(config.vlm_config, text_config, vision_config)
        if config.num_vlm_layers < text_config_full.num_hidden_layers:
            self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[: config.num_vlm_layers]
        self.config = config.vlm_config

        expert_hidden = config.expert_hidden_size
        expert_text_cfg = _expert_text_config(text_config, expert_hidden, config.n_expert_layers)
        self.lm_expert = ExpertModel(config, expert_text_cfg, expert_hidden)

        self.num_attention_heads = text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.attention = EagerAttention(
            text_config.num_attention_heads, text_config.num_key_value_heads, text_config.head_dim
        )

        self.processor = tokenizer
        # Per-request mutable state, shared with ``VLAFlowMatching``: prefix length ->
        # (KVCache, fill_count, {cross layer: (k, v)}), valid only for that exact fill.
        self.expert_prefix_kv: dict[int, tuple[KVCache, int, dict[int, tuple[torch.Tensor, torch.Tensor]]]] | None = (
            None
        )
        # Drift tier: fused Triton GQA kernels replace the denoising self-/cross-
        # attention chains, so numbers differ from the fp32 eager path by design.
        self.triton_attn = False
        # Drift tier: fused kernels replace the LLM prefill attention chain
        # (input_layernorm + q/k/v + RoPE + KV fill + GQA attention).
        self.fused_prefill_attn = False

    def _cached_expert_prefix_kv(self, past_key_values, layer_idx: int):
        """Return the cached expert prefix K/V for a cross layer, else ``None``."""
        if not self.expert_prefix_kv:
            return None
        entry = self.expert_prefix_kv.get(past_key_values.prefix_len)
        if entry is None:
            return None
        kv, fill_count, layer_cache = entry
        if kv is not past_key_values or fill_count != past_key_values.fill_count:
            return None
        return layer_cache.get(layer_idx)

    # ------------------------------------------------------------------ #
    # embeddings
    # ------------------------------------------------------------------ #
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        vision_model = self.vlm.model.vision_model
        image_hidden_states = vision_model(image.to(dtype=vision_model.embeddings.patch_embedding.weight.dtype))
        image_hidden_states = self.vlm.model.connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vlm.model.text_model.embed_tokens(tokens)

    # ------------------------------------------------------------------ #
    # layer dispatch
    # ------------------------------------------------------------------ #
    def get_model_layers(self):
        vlm_layers = list(self.vlm.model.text_model.layers)
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = self.lm_expert.layers[expert_layer_index]
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        use_cache: bool | None = True,
        fill_kv_cache: bool = False,
        past_key_values: KVCache | None = None,
    ):
        query_states, key_states, value_states = [], [], []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            # Fused LLM prefill layer (drift tier): replaces input_layernorm + q/k/v +
            # RoPE + KV fill + GQA attention; the eager path below stays untouched.
            if (
                self.fused_prefill_attn
                and fill_kv_cache
                and i == 0
                and (len(inputs_embeds) == 1 or inputs_embeds[1] is None)
            ):
                from .triton_attn import triton_prefill_layer

                seq_len = hidden_states.shape[1]
                att_output = triton_prefill_layer(
                    hidden_states.contiguous(),
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                    position_ids[:, :seq_len],
                    attention_mask[:, :seq_len, :seq_len][0],
                    past_key_values,
                    layer_idx,
                    layer.self_attn.head_dim,
                )
                return [att_output], past_key_values
            # Fused GQA self-attention (drift tier) for the denoising path -- the only
            # path with ``fill_kv_cache=False`` and a suffix input.
            if self.triton_attn and not fill_kv_cache and i == 1:
                from .triton_attn import triton_self_attn

                # Raw (pre-norm) fp32 hidden as-is: the kernel RMSNorms in fp32.
                prefix_len = past_key_values.prefix_len
                prefix_pad = attention_mask[0, 0, :prefix_len]  # bool [P]
                suffix_pos = position_ids[0]  # [S] denoising positions
                prefix_k, prefix_v = past_key_values.view(layer_idx, 0, prefix_len)
                att_output = triton_self_attn(
                    hidden_states[0],
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                    prefix_k[0],
                    prefix_v[0],
                    suffix_pos,
                    prefix_pad,
                    layer.self_attn.head_dim,
                )
                return [att_output.unsqueeze(0)], past_key_values
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # [B, total_seq, heads, head_dim]
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        query_states = apply_rope(query_states, _position_ids)
        key_states = apply_rope(key_states, _position_ids)

        if use_cache:
            if fill_kv_cache:
                past_key_values.fill(layer_idx, key_states, value_states)
            else:
                # Suffix K/V goes right after the prefix and is read back as one
                # [prefix; suffix] view: bit-identical to torch.cat, no per-step allocation.
                past_key_values.write_suffix(layer_idx, key_states, value_states)
                key_states, value_states = past_key_values.view(
                    layer_idx, 0, past_key_values.prefix_len + key_states.shape[1]
                )

        att_output = self.attention(_attention_mask, batch_size, query_states, key_states, value_states)
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        use_cache: bool | None = True,
        fill_kv_cache: bool = False,
        past_key_values: KVCache | None = None,
    ):
        att_outputs = []

        # Cross layers never fill the cache: prefix prefill stored the post-RoPE VLM
        # K/V of every layer through ``forward_attn_layer``; here we only read it back.
        key_states = past_key_values[layer_idx]["key_states"]
        value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            # Expert prefix K/V is loop-invariant: reuse the cached projection, else
            # project inline with the same ops (bit-identical to the cached path).
            cached = self._cached_expert_prefix_kv(past_key_values, layer_idx)
            if cached is not None:
                expert_key_states, expert_value_states = cached
            else:
                _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                    *key_states.shape[:2], -1
                )
                expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                    *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
                )

                _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                    *value_states.shape[:2], -1
                )
                expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                    *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
                )

            # Fused GQA cross-attention (drift tier) for the denoising path, and for every
            # step-0 layer too -- i.e. even before the expert prefix-KV cache exists.
            if self.triton_attn and not fill_kv_cache:
                from .triton_attn import triton_cross_attn

                hidden2d = expert_hidden_states[0].contiguous()
                expert_position_id = position_ids - torch.min(position_ids, dim=1, keepdim=True).values
                prefix_pad = attention_mask[0, 0, : expert_key_states.shape[1]]  # bool [P]
                att_output = triton_cross_attn(
                    hidden2d,
                    expert_layer.input_layernorm.weight,
                    expert_layer.input_layernorm.variance_epsilon,
                    expert_layer.self_attn.q_proj.weight,
                    expert_key_states[0],
                    expert_value_states[0],
                    expert_position_id[0],
                    prefix_pad,
                    expert_layer.self_attn.head_dim,
                )
                att_outputs.append(att_output.unsqueeze(0))
                return att_outputs, past_key_values

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            expert_position_id = position_ids - torch.min(position_ids, dim=1, keepdim=True).values  # start from 0
            expert_attention_mask = attention_mask[:, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1]]

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = self.attention(
                expert_attention_mask,
                batch_size,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    # ------------------------------------------------------------------ #
    # main forward
    # ------------------------------------------------------------------ #
    def _forward_one_layer(
        self,
        model_layers,
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids,
        attention_mask,
        batch_size,
        use_cache: bool | None = True,
        fill_kv_cache: bool = False,
        past_key_values: KVCache | None = None,
    ):
        """Run one layer plus the residual/MLP plumbing; return the next embeddings.

        ``prefill_layer`` and ``step0_layer`` drive single layers through it.
        """
        if (
            fill_kv_cache
            or "cross" not in self.attention_mode
            or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
        ):
            att_outputs, past_key_values = self.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )
        else:
            att_outputs, past_key_values = self.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )
        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
            if hidden_states is not None:
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue
                end = start + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)

                # In-place add: the fp32 prefix embedding is downcast here on the first
                # residual -- reproduce the ``+=`` semantics, do not "fix" the dtype.
                out_emb += hidden_states
                after_first_residual = out_emb.clone()

                if getattr(layer.mlp, "fused_norm_gate_up", False):
                    # Fused post-attention RMSNorm + gate/up + silu (drift tier).
                    from .triton_mlp import triton_norm_gate_up

                    emb_shape = out_emb.shape
                    act = triton_norm_gate_up(
                        out_emb.reshape(-1, emb_shape[-1]),
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.variance_epsilon,
                        layer.mlp.gate_proj.weight,
                        layer.mlp.up_proj.weight,
                    )
                    out_emb = layer.mlp.down_proj(act).reshape(emb_shape)
                else:
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                out_emb += after_first_residual

                outputs_embeds.append(out_emb)

                start = end if len(att_outputs) == 1 else 0
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def prefill_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids,
        attention_mask,
        batch_size: int,
        past_key_values: KVCache,
    ):
        """Run one LLM prefill layer: fill prefix KV[layer_idx] (post-RoPE) and return
        the next prefix hidden state.

        Must be a pure function of its inputs and ``past_key_values``: the fused
        two-stream branch relies on that.
        """
        outputs, past_key_values = self._forward_one_layer(
            self.get_model_layers(),
            [hidden_states, None],
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            use_cache=True,
            fill_kv_cache=True,
            past_key_values=past_key_values,
        )
        return outputs[0], past_key_values

    def step0_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids,
        attention_mask,
        batch_size: int,
        past_key_values: KVCache,
    ):
        """Run one expert layer of denoising step 0 (self-attn even, cross-attn odd).

        Cross layers project the prefix K/V inline when the post-prefill cache is
        not populated yet -- bit-identical to the cached path.
        """
        outputs, past_key_values = self._forward_one_layer(
            self.get_model_layers(),
            [None, hidden_states],
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            use_cache=True,
            fill_kv_cache=False,
            past_key_values=past_key_values,
        )
        return outputs[1], past_key_values

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: list[torch.Tensor | None] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        models = [self.vlm.model.text_model, self.lm_expert]
        model_layers = self.get_model_layers()
        batch_size = 0
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]
            break

        fill_kv_cache = bool(fill_kv_cache)
        if fill_kv_cache and past_key_values is None:
            # Fresh cache: buffers are allocated lazily on the first fill below.
            past_key_values = KVCache(
                num_layers=self.num_vlm_layers,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
            )

        num_layers = self.num_vlm_layers
        for layer_idx in range(num_layers):
            inputs_embeds, past_key_values = self._forward_one_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values


def _expert_text_config(text_config, expert_hidden_size: int, num_layers: int):
    """Copy the text config with the expert's hidden/intermediate sizes."""
    from dataclasses import replace

    intermediate = get_intermediate_size(expert_hidden_size)
    return replace(
        text_config,
        hidden_size=expert_hidden_size,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
    )


def get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: int = 4, multiple_of: int = 256) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
