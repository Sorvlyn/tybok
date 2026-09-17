"""SmolVLA task tokenization (GPT-2 style byte-level BPE).

SmolVLM2 uses a GPT-2 style byte-level BPE loaded from the serialized
``tokenizer.json`` via the standalone ``tokenizers`` library. Contract:

    tokenizer(text, max_length=48, truncation=True, padding="max_length",
              padding_side="right", return_tensors="pt")

which produces ``input_ids`` (1, 48) long and a boolean ``attention_mask`` (1, 48).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class TokenizerSpec:
    """Token ids needed by the model."""

    pad_token_id: int
    fake_image_token_id: int
    global_image_token_id: int
    image_token_id: int
    bos_token_id: int
    eos_token_id: int


class TaskTokenizer:
    """Wraps a ``tokenizers.Tokenizer`` with right-padded truncation to ``max_length``."""

    def __init__(self, tokenizer_dir: str, max_length: int = 48):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.max_length = max_length
        # truncation + right padding to max_length
        self.tokenizer.enable_truncation(max_length=max_length)
        self.spec = self._build_spec(tokenizer_dir)

    def _build_spec(self, tokenizer_dir: str) -> TokenizerSpec:
        import json

        # pad token is "<|im_end|>" (id 2) for SmolVLM2; fall back to config
        pad_token_id = self.tokenizer.token_to_id("<|im_end|>")
        if pad_token_id is None:
            with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
                tcfg = json.load(f)
            pad_token_id = self.tokenizer.token_to_id(tcfg.get("pad_token", "<|im_end|>"))

        fake_id = self.tokenizer.token_to_id("<fake_token_around_image>")
        global_id = self.tokenizer.token_to_id("<global-img>")
        image_id = self.tokenizer.token_to_id("<image>")
        bos_id = self.tokenizer.token_to_id("<|im_start|>")
        eos_id = self.tokenizer.token_to_id("<end_of_utterance>")
        if fake_id is None:
            fake_id = self.tokenizer.token_to_id("<fake_token_around_image>")
        return TokenizerSpec(
            pad_token_id=pad_token_id,
            fake_image_token_id=fake_id,
            global_image_token_id=global_id,
            image_token_id=image_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
        )

    def encode_task(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a task string (right-padded to ``max_length``).

        Returns:
            input_ids: long tensor (1, max_length)
            attention_mask: bool tensor (1, max_length)
        """
        # right padding; pad id / token come from the tokenizer spec
        self.tokenizer.enable_padding(
            pad_id=self.spec.pad_token_id,
            pad_token="<|im_end|>",
            length=self.max_length,
            direction="right",
        )
        encoding = self.tokenizer.encode(task)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long)
        attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.bool)
        return input_ids, attention_mask
