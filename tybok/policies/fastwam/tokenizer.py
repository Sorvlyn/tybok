"""fastWAM task tokenization (UMT5).

Loads ``tokenizer.json`` with the standalone ``tokenizers`` library and applies
the same call as the reference (``padding="max_length"``, truncation,
``add_special_tokens=True``): the post-processor appends ``</s>`` (id 1),
yielding ``input_ids`` (1, max_len) plus an ``attention_mask``.
"""

from __future__ import annotations

import os

import torch


class UMT5Tokenizer:
    """Wraps a ``tokenizers.Tokenizer`` with the FastWAM-compatible call."""

    def __init__(self, tokenizer_dir: str, max_length: int = 128):
        from tokenizers import Tokenizer

        if not os.path.isdir(tokenizer_dir):
            raise ValueError(f"tokenizer dir {tokenizer_dir!r} does not exist")
        self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.max_length = int(max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one prompt the way ``WanTokenizer`` does for the text encoder.

        Returns:
            input_ids: long tensor (1, max_length)
            attention_mask: bool tensor (1, max_length)  -- True for non-pad
        """
        self.tokenizer.enable_padding(
            pad_id=self.pad_token_id,
            pad_token="<pad>",
            length=self.max_length,
            direction="right",
        )
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long)
        attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.bool)
        return input_ids, attention_mask
