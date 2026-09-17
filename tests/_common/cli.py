"""The command line surface every check shares.

``--level``, ``--device``, ``--checkpoint``, ``--require-gpu`` and ``--list-jobs`` mean the same
thing in every check; ``--job-index`` is the internal hook a driver uses to run exactly one job
(see :mod:`tests._common.process`). Missing environment (no CUDA / no checkpoint) is reported as
``SKIP`` with exit code 0 so CPU-only CI can run the same command; ``--require-gpu`` turns that
into a failure for machines that must actually run the GPU path.

Checkpoints are never hard-coded: a model's default comes from ``TYBOK_CHECKPOINT_<MODEL>``,
which is also read from ``tests/checkpoints.env`` when that file exists (see
``tests/checkpoints.env.example``). ``--checkpoint`` overrides it for one run.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .level import Job, Level

__all__ = [
    "CHECKPOINT_ENV_PREFIX",
    "ENV_FILE",
    "ModelSpec",
    "add_engine_arguments",
    "add_level_arguments",
    "check_description",
    "checkpoint_from_env",
    "cuda_available",
    "job_keys",
    "load_env_file",
    "missing_environment",
    "print_job_list",
]


@dataclass(frozen=True)
class ModelSpec:
    """Everything a check needs to know about the model it exercises."""

    key: str
    """Model key as ``create_engine`` knows it (``fastwam`` / ``pi05`` / ``smolvla``)."""

    checkpoint: str
    """Default checkpoint directory: ``TYBOK_CHECKPOINT_<MODEL>``, overridable with ``--checkpoint``."""

    fused_flags: tuple[str, ...] = ()
    """Fused (Triton) tier for this model, as ``create_engine`` keyword flags."""


CHECKPOINT_ENV_PREFIX = "TYBOK_CHECKPOINT_"
"""Environment variable holding a model's default checkpoint: ``TYBOK_CHECKPOINT_<MODEL>``."""

ENV_FILE = Path(__file__).resolve().parents[1] / "checkpoints.env"
"""Optional ``KEY=VALUE`` file loaded into the environment (``tests/checkpoints.env``)."""

_env_loaded = False


def load_env_file(path: Path | None = None) -> None:
    """Load ``KEY=VALUE`` lines from ``path`` (default :data:`ENV_FILE`) into ``os.environ``.

    Exported variables win, so an already-set ``TYBOK_CHECKPOINT_<MODEL>`` is not overwritten, and
    a missing file is fine: the checks then report ``SKIP`` until a checkpoint is passed.

    Args:
        path: file to read; :data:`ENV_FILE` when omitted. Read once per process.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        lines = (ENV_FILE if path is None else path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("\"'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), os.path.expanduser(value))


def checkpoint_from_env(model_key: str) -> str:
    """Default checkpoint directory for ``model_key``, or ``""`` when it is not configured.

    Args:
        model_key: model key as ``create_engine`` knows it (``smolvla`` / ``pi05`` / ``fastwam``).

    Returns:
        ``$TYBOK_CHECKPOINT_<MODEL>``, after :func:`load_env_file`; empty string when unset.
    """
    load_env_file()
    return os.environ.get(f"{CHECKPOINT_ENV_PREFIX}{model_key.upper()}", "")


def add_level_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--level`` / ``--job-index`` / ``--list-jobs`` -- what every check needs."""
    parser.add_argument(
        "--level",
        type=Level,
        choices=("quick", "full"),
        metavar="{quick,full}",
        default=Level.QUICK,
        help="quick = the everyday subset; full = the regression run (default: quick)",
    )

    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="internal: run the single job with this index and print its row",
    )

    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="print the jobs the current --level runs, then exit",
    )


def add_engine_arguments(
    parser: argparse.ArgumentParser,
    spec: ModelSpec,
    *,
    device: str = "cuda",
    with_require_gpu: bool = True,
) -> None:
    """Add the arguments of a check that builds engines: level, device, checkpoint, skipping."""
    add_level_arguments(parser)
    parser.add_argument("--device", default=device, help="torch device for the engine (default: cuda)")
    parser.add_argument(
        "--checkpoint",
        default=spec.checkpoint,
        help=f"{spec.key} checkpoint directory (default: "
        f"${CHECKPOINT_ENV_PREFIX}{spec.key.upper()} or tests/checkpoints.env)",
    )

    if with_require_gpu:
        parser.add_argument(
            "--require-gpu",
            action="store_true",
            help="FAIL instead of SKIP when CUDA or the checkpoint is unavailable",
        )


def check_description(doc: str | None) -> str:
    """First paragraph of a check's module docstring, for ``argparse``'s ``description``."""
    paragraph: list[str] = []
    for line in (doc or "").strip().splitlines():
        if not line.strip():
            break
        paragraph.append(line.strip())
    return " ".join(paragraph)


def cuda_available(device: str) -> bool:
    """Whether ``device`` can be used on this machine (a CPU device is always available)."""
    if not device.startswith("cuda"):
        return True
    import torch

    return bool(torch.cuda.is_available())


def missing_environment(device: str, checkpoint: str) -> str | None:
    """Reason to skip (no CUDA / no checkpoint), or ``None`` when the check can run."""
    if not cuda_available(device):
        return f"CUDA unavailable on {device}"
    if not checkpoint:
        return "no checkpoint configured (--checkpoint or TYBOK_CHECKPOINT_<MODEL>)"
    if not os.path.isdir(checkpoint):
        return f"checkpoint not found: {checkpoint}"
    return None


def print_job_list(job_keys: Sequence[str]) -> None:
    """Print the jobs of the current level, one per line, with their indexes."""
    width = len(str(len(job_keys) - 1)) if job_keys else 1
    for index, key in enumerate(job_keys):
        print(f"{index:>{width}}  {key}")


def job_keys(jobs: Iterable[Job]) -> list[str]:
    """Display keys of ``jobs``, in the order the driver will run them."""
    return [job.key for job in jobs]
