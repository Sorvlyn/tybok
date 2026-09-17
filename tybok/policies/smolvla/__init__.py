"""SmolVLA policy backend.

Deployment engine (SigLIP vision encoder + Gemma text decoder + action expert +
flow-matching denoising), registered as ``model_type="smolvla"``.

``engine.py`` (engine) / ``config.py`` (checkpoint + VLM config) /
``tokenizer.py`` (task tokenization) / ``preprocess.py`` (pre/post + stats) /
``models/`` (PyTorch ``nn.Module`` code).
"""

from .engine import SmolVLAEngine, print_validation_report

__all__ = ["SmolVLAEngine", "print_validation_report"]
