#!/usr/bin/env python3
"""pi05 drift check: ``eager+graph`` / ``fused+graph`` against the eager reference.

Each arm runs in its own child process: a captured CUDA graph owns a private memory
pool and keeps every tensor it reads alive for its whole lifetime, so two arms cannot
share one process. All arms run the *same* synthetic frame with the *same* denoising
noise, so any output difference comes from the arm itself.

Expected results:

* ``eager`` -- the reference: no CUDA graph, no fused kernels.
* ``eager+graph`` -- bit-exact against the reference; a CUDA graph only changes how
  the identical kernels are launched.
* ``fused+graph`` -- small bounded drift; the fused Triton tiers reassociate the
  arithmetic inside the ViT attention, the LLM attention and the expert MLP
  (``--fp8`` adds the W8A8 fp8 MLP chain). The math is equivalent, the bits are not.

Usage::

    python dev/pi05_drift_check.py                       # $TYBOK_CHECKPOINT_PI05
    python dev/pi05_drift_check.py --checkpoint /path/to/pi05 --cameras 2 --steps 10
    python dev/pi05_drift_check.py --noise-zero                  # deterministic sampler math
    python dev/pi05_drift_check.py --arms eager+graph            # graph only, no fused tier

Exit code is 0 when every non-fused arm is bit-exact against the reference and every
fused arm stays inside ``--tolerance``, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ENV_FILE = os.path.join(PROJECT_ROOT, "tests", "checkpoints.env")
ROW_PREFIX = "TYBOK_DRIFT_ROW "

REFERENCE_ARM = "eager"
EXPERIMENT_ARMS = ("eager+graph", "fused+graph")

# The documented best-performance tier for pi05; --fp8 adds the W8A8 MLP chain on top.
FUSED_FLAGS = ("tl_fused_vit", "tl_llm_flash_attn", "tl_fused_expert")
FP8_FLAGS = ("tl_fp8_llm_mlp", "tl_fp8_expert_mlp")


def default_checkpoint(model: str) -> str:
    """``$TYBOK_CHECKPOINT_<MODEL>``, else the value in ``tests/checkpoints.env``, else ``""``."""
    key = f"TYBOK_CHECKPOINT_{model.upper()}"
    if os.environ.get(key):
        return os.environ[key]
    try:
        with open(ENV_FILE, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


def _count_graphs(obj) -> int:
    """Number of ``torch.cuda.CUDAGraph`` objects reachable from a graph-runner entry."""
    import torch

    if isinstance(obj, torch.cuda.CUDAGraph):
        return 1
    if isinstance(obj, dict):
        return sum(_count_graphs(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_graphs(value) for value in obj)
    return 0


def run_arm(arm: str, args: argparse.Namespace) -> dict:
    """Child side: build one arm's engine, run one request, return its row."""
    import numpy as np
    import torch

    from tybok.registry import create_engine

    parts = arm.split("+")
    graph = "graph" in parts
    fused = "fused" in parts
    flags = {name: True for name in FUSED_FLAGS} if fused else {}
    if fused and args.fp8:
        flags.update({name: True for name in FP8_FLAGS})

    t0 = time.perf_counter()
    engine = create_engine(
        args.checkpoint,
        model_type=args.model,
        device=args.device,
        graph=graph,
        graph_cameras=args.graph_cameras,
        num_steps=args.steps,
        pad_free=args.pad_free,
        warmup=True,
        **flags,
    )
    setup_s = time.perf_counter() - t0

    spec = engine.describe()
    frame = engine._profile_frame(args.cameras, args.frame_seed)
    if args.noise_zero:
        noise = torch.zeros(1, spec["chunk_size"], spec["action_dim"])
    else:
        noise = torch.randn(
            1,
            spec["chunk_size"],
            spec["action_dim"],
            generator=torch.Generator().manual_seed(args.noise_seed),
        )
    chunk = np.asarray(engine.predict_action_chunk(frame, noise=noise), dtype=np.float64)

    runner = getattr(engine, "_graph_runner", None)
    graphs = sum(_count_graphs(entry) for entry in runner.entries.values()) if runner is not None else 0

    return {
        "arm": arm,
        "graph": int(graph),
        "fused": int(fused),
        "graphs": graphs,
        "setup_s": setup_s,
        "chunk_size": int(spec["chunk_size"]),
        "action_dim": int(spec["action_dim"]),
        "values": [float(value) for value in chunk.ravel()],
    }


def spawn(arm: str, argv: list[str]) -> dict:
    """Run one arm in a child process and return its row (an error row when it dies)."""
    command = [sys.executable, os.path.abspath(__file__), *argv, "--arm", arm]
    completed = subprocess.run(command, capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        if line.startswith(ROW_PREFIX):
            return json.loads(line[len(ROW_PREFIX) :])
    tail = (completed.stderr.strip().splitlines() or ["<no stderr>"])[-3:]
    return {"arm": arm, "error": f"child exited {completed.returncode}: " + " | ".join(tail)}


def measure(reference: list[float], values: list[float]) -> tuple[float, float]:
    """``(max abs difference, rms difference)`` between two flattened chunks."""
    import numpy as np

    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(values, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return float(np.abs(diff).max()), float(np.sqrt(np.mean(diff * diff)))


def verdict(row: dict, reference: list[float], tolerance: float) -> tuple[bool, str, float, float]:
    """Judge one arm against the reference chunk; returns (ok, verdict, max_abs, rms)."""
    if "error" in row:
        return False, f"FAILED ({row['error']})", float("nan"), float("nan")
    max_abs, rms = measure(reference, row["values"])
    if row["arm"] == REFERENCE_ARM:
        return True, "reference", max_abs, rms
    if max_abs == 0.0:
        return True, "bit-exact", max_abs, rms
    if row["fused"]:
        ok = max_abs <= tolerance
        label = f"drift {'within' if ok else 'ABOVE'} tolerance ({tolerance:.1e})"
        return ok, label, max_abs, rms
    # A non-fused arm must reproduce the reference exactly: a graph is a launch change only.
    return False, "NOT bit-exact", max_abs, rms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="pi05 checkpoint directory (default: $TYBOK_CHECKPOINT_PI05 or tests/checkpoints.env)",
    )
    parser.add_argument("--model", default="pi05", help="registry key of the engine to build")
    parser.add_argument("--device", default="cuda", help="torch device for the engine")
    parser.add_argument(
        "--arms",
        default=",".join(EXPERIMENT_ARMS),
        help=f"comma-separated experiment arms ({'|'.join(EXPERIMENT_ARMS)}); "
        f"'{REFERENCE_ARM}' is always run as the reference",
    )
    parser.add_argument("--cameras", type=int, default=None, help="cameras in the test frame (default: all)")
    parser.add_argument("--steps", type=int, default=None, help="denoising steps (default: checkpoint config)")
    parser.add_argument(
        "--graph-cameras",
        default=None,
        help="comma-separated camera counts to pre-capture, e.g. '2,3' (requires --pad-free)",
    )
    parser.add_argument(
        "--pad-free",
        action="store_true",
        help="skip empty camera slots in the graph (bucket the graph by real camera count)",
    )
    parser.add_argument("--fp8", action="store_true", help="add the W8A8 fp8 MLP tiers to the fused arm")
    parser.add_argument("--noise-seed", type=int, default=123, help="seed of the denoising noise")
    parser.add_argument("--noise-zero", action="store_true", help="use zero noise (isolates the sampler math)")
    parser.add_argument("--frame-seed", type=int, default=1, help="seed of the synthetic test frame")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-2,
        help="max abs drift allowed for a fused arm (default: 5e-2, the documented chunk tolerance)",
    )
    parser.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.arm:
        try:
            row = run_arm(args.arm, args)
        except Exception as error:  # noqa: BLE001 - the driver reports it as a failed row
            row = {"arm": args.arm, "error": f"{type(error).__name__}: {error}"}
        print(ROW_PREFIX + json.dumps(row), flush=True)
        return 0

    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.model)
    if not args.checkpoint:
        raise SystemExit("no checkpoint: pass --checkpoint or set TYBOK_CHECKPOINT_PI05")
    if not os.path.isdir(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.graph_cameras:
        args.graph_cameras = tuple(int(part) for part in args.graph_cameras.split(",") if part.strip())

    arms = [part.strip() for part in args.arms.split(",") if part.strip()]
    unknown = [arm for arm in arms if arm not in EXPERIMENT_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {EXPERIMENT_ARMS}")
    arms = [REFERENCE_ARM, *arms]

    if not args.device.startswith("cuda"):
        raise SystemExit("pi05 arms need CUDA (the graph path and the fused tiers are CUDA-only)")

    import torch

    print(f"torch {torch.__version__} | device {args.device} | checkpoint {args.checkpoint}")
    print(
        f"frame seed {args.frame_seed} | cameras {args.cameras or 'all'} | steps {args.steps or 'config'} "
        f"| noise {'zero' if args.noise_zero else f'seed={args.noise_seed}'}"
    )
    print(f"gpu {torch.cuda.get_device_name(0)}")

    argv = list(sys.argv[1:])
    rows = {}
    for arm in arms:
        print(f"[drift] running arm '{arm}' ...", flush=True)
        rows[arm] = spawn(arm, argv)

    reference = rows[REFERENCE_ARM]
    if "error" in reference:
        raise SystemExit(f"reference arm failed: {reference['error']}")

    verdicts = [(arm, *verdict(rows[arm], reference["values"], args.tolerance)) for arm in arms]

    # |reference| scale, so the absolute drift can be read relative to the chunk itself.
    import numpy as np

    scale = float(np.abs(np.asarray(reference["values"], dtype=np.float64)).max())

    print(
        f"\n=== pi05 drift vs '{REFERENCE_ARM}' (chunk {reference['chunk_size']}x{reference['action_dim']}, "
        f"max|ref| {scale:.3f}) ==="
    )
    print(
        f"{'arm':<14} {'graph':>5} {'fused':>5} {'graphs':>6} {'setup(s)':>8} {'max|d|':>10} {'rms|d|':>10} "
        f"{'rel|d|':>10}  verdict"
    )
    for arm, _ok, label, max_abs, rms in verdicts:
        row = rows[arm]
        if "error" in row:
            print(f"{arm:<14} {'-':>5} {'-':>5} {'-':>6} {'-':>8} {'-':>10} {'-':>10} {'-':>10}  {label}")
            continue
        rel = max_abs / scale if scale else 0.0
        print(
            f"{arm:<14} {row['graph']:>5} {row['fused']:>5} {row['graphs']:>6} {row['setup_s']:>8.1f} "
            f"{max_abs:>10.2e} {rms:>10.2e} {rel:>10.2e}  {label}"
        )

    failed = [arm for arm, ok, *_ in verdicts if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({', '.join(failed)})")
        return 1
    print(f"RESULT: PASS ({', '.join(arms)}; non-fused arms bit-exact, fused drift within {args.tolerance:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
