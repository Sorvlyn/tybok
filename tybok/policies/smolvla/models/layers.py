"""Low-level building blocks (norms, activations, projections).

SmolVLM text decoder: Llama/Gemma RMSNorm. SigLIP vision encoder: LayerNorm
(with bias) and ``gelu_pytorch_tanh``. Thin ``torch`` wrappers so the checkpoint
can be loaded by name.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Llama-style RMSNorm (fp32 accumulation, cast back to input dtype)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNormV2(nn.Module):
    """SmolVLM variant of RMSNorm (identical math)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SiLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


class GELUTanh(nn.Module):
    """gelu_pytorch_tanh: GELU with the tanh approximation (SigLIP MLP)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate="tanh")


class MLP(nn.Module):
    """Standard decoder MLP: gate/up (SiLU) + down, or fc1/fc2 for the vision MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiLUActivation()
        assert activation == "silu"
        # ``--tl-fused-expert``: fuse the post-attention RMSNorm + gate/up + silu into
        # one kernel; down_proj stays cuBLAS.
        self.fused_norm_gate_up = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class VisionMLP(nn.Module):
    """SigLIP-style MLP: fc1 -> gelu_pytorch_tanh -> fc2 (all with bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = GELUTanh()
        assert activation == "gelu_pytorch_tanh"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
