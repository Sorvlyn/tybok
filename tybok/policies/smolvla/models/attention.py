"""Eager attention + static KV buffers for the SmolVLA expert decoder.

Attention upcasts q/k to float32, applies the boolean 2-D mask with a large
negative constant, softmaxes in fp32 and multiplies by the (possibly bf16)
values; scaling is ``head_dim**-0.5``.

KV cache layout is per-layer ``[B, L, Hkv, D]`` (seq-major) in **preallocated
contiguous buffers** sized for ``max_prefix_len + suffix_len``. The prefix K/V is
copied into the buffer head once at prefill and the denoising suffix is
overwritten in place every step, so self-attention reads ``[prefix; suffix]``
without per-step ``torch.cat`` (bit-exact vs concatenation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KVCache:
    """Static per-layer KV buffers: ``[num_layers, B, L, Hkv, D]``.

    ``fill`` writes the prefix K/V of a layer into the buffer head;
    ``write_suffix`` overwrites the region right after the prefix with the
    denoising suffix; ``view`` returns ``[B, L, Hkv, D]`` slices so callers
    attend over ``[prefix; suffix]`` without concatenating. ``__getitem__``
    keeps dict-style access returning the *prefix* slice (cross-attention and
    validation code).

    Buffers allocate lazily on the first ``fill``. In the CUDA-graph path the
    cache is created once with ``suffix_len = chunk_size`` before capture so the
    addresses stay stable across replays; eager/validate paths may grow.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int = 1,
        suffix_len: int = 0,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.suffix_len = suffix_len
        self.prefix_len = 0
        # Monotonic counter bumped on every ``fill`` (prefill); stored by the
        # expert prefix-KV cache so any re-fill or different cache object
        # invalidates the cached projections automatically.
        self.fill_count = 0
        self.key_buf: torch.Tensor | None = None
        self.value_buf: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def _allocate(self, key: torch.Tensor) -> None:
        total = key.shape[1] + self.suffix_len
        shape = (self.num_layers, key.shape[0], total, self.num_kv_heads, self.head_dim)
        self.key_buf = torch.empty(shape, dtype=key.dtype, device=key.device)
        self.value_buf = torch.empty(shape, dtype=key.dtype, device=key.device)

    def _grow(self, need: int, dtype: torch.dtype, device: torch.device) -> None:
        """Reallocate the buffers with room for ``need`` seq positions.

        Only reached outside the CUDA-graph path (where sizes are fixed and the
        captured addresses must stay stable).
        """
        shape = (self.num_layers, self.batch_size, need, self.num_kv_heads, self.head_dim)
        new_key = torch.empty(shape, dtype=dtype, device=device)
        new_value = torch.empty(shape, dtype=dtype, device=device)
        if self.key_buf is not None:
            n = min(self.key_buf.shape[2], need)
            new_key[:, :, :n].copy_(self.key_buf[:, :, :n])
            new_value[:, :, :n].copy_(self.value_buf[:, :, :n])
        self.key_buf = new_key
        self.value_buf = new_value

    def fill(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the prefix K/V of ``layer_idx`` (first ``key.shape[1]`` positions)."""
        self.prefix_len = key.shape[1]
        self.fill_count += 1
        if self.key_buf is None or key.shape[1] > self.key_buf.shape[2] - self.suffix_len:
            self._allocate(key)
        assert self.key_buf is not None and self.value_buf is not None
        self.key_buf[layer_idx, :, : self.prefix_len].copy_(key)
        self.value_buf[layer_idx, :, : self.prefix_len].copy_(value)

    def write_suffix(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Overwrite the suffix region right after the prefix in place."""
        need = self.prefix_len + key.shape[1]
        if self.key_buf is None or need > self.key_buf.shape[2]:
            self._grow(need, key.dtype, key.device)
        assert self.key_buf is not None and self.value_buf is not None
        self.key_buf[layer_idx, :, self.prefix_len : need].copy_(key)
        self.value_buf[layer_idx, :, self.prefix_len : need].copy_(value)

    def view(self, layer_idx: int, start: int = 0, end: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B, end-start, Hkv, D]`` slices of the layer's buffers."""
        if end is None:
            end = self.prefix_len
        assert self.key_buf is not None and self.value_buf is not None, "KVCache used before fill()"
        return self.key_buf[layer_idx, :, start:end], self.value_buf[layer_idx, :, start:end]

    def __getitem__(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Dict-compatible prefix access (cross-attn layers + validation code)."""
        k, v = self.view(layer_idx, 0, self.prefix_len)
        return {"key_states": k, "value_states": v}


class EagerAttention(nn.Module):
    """Multi-head eager attention with GQA (fp32).

    Upcasts q/k to float32, applies the boolean 2-D mask, softmaxes in fp32 and
    multiplies by the values; bit-exact vs the reference eager implementation.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(
        self,
        attention_mask: torch.Tensor,  # bool [B, Lq, Lk]
        batch_size: int,
        query_states: torch.Tensor,  # [B, Lq, Hq, D]
        key_states: torch.Tensor,  # [B, Lk, Hkv, D]
        value_states: torch.Tensor,  # [B, Lk, Hkv, D]
    ) -> torch.Tensor:
        return self._forward_eager(attention_mask, batch_size, query_states, key_states, value_states)

    def _forward_eager(
        self,
        attention_mask: torch.Tensor,  # bool [B, Lq, Lk]
        batch_size: int,
        query_states: torch.Tensor,  # [B, Lq, H, D]
        key_states: torch.Tensor,  # [B, Lk, Hkv, D]
        value_states: torch.Tensor,  # [B, Lk, Hkv, D]
    ) -> torch.Tensor:
        head_dim = self.head_dim
        num_key_value_heads = self.num_kv_heads
        num_att_heads = self.num_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = F.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
