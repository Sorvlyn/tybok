"""pi0.5 task prompt tokenization (PaliGemma tokenizer).

Loads the PaliGemma ``tokenizer.json`` with the standalone ``tokenizers``
library and applies the reference settings (truncate to ``max_length=200``,
right padding), producing ``input_ids`` (1, 200) with a leading ``<bos>``
(id 2) and a boolean ``attention_mask`` (1, 200).
"""

from __future__ import annotations

import os

import torch


class PaliGemmaTokenizer:
    """Wraps a ``tokenizers.Tokenizer`` with the lerobot-compatible call."""

    def __init__(self, tokenizer_dir: str, max_length: int = 200):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.max_length = max_length
        # Reference settings: truncation, right padding to max_length. The
        # tokenizer.json post-processor prepends <bos> (id 2).
        self.tokenizer.enable_truncation(max_length=max_length)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_token_id is None:
            self.pad_token_id = 0
        self.bos_token_id = self.tokenizer.token_to_id("<bos>")

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a prompt into padded ids / mask.

        Returns:
            input_ids: long tensor (1, max_length)
            attention_mask: bool tensor (1, max_length)
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
