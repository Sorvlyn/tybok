#!/usr/bin/env python3
"""pi05 inference speed test: ``eager+graph`` vs ``fused+graph``.

Measures per-request latency of ``predict_action_chunk`` (mean / p50 / p95 / min,
Hz) for each arm and prints a comparison table, plus the fused-to-eager ratio.
Each arm runs in its own child process and its own engine: a captured CUDA graph
owns a private memory pool and keeps every tensor it reads alive, so the arms
cannot share one process, and weights stay resident for only one arm at a time.

Usage::

    python dev/pi05_bench.py                             # $TYBOK_CHECKPOINT_PI05
    python dev/pi05_bench.py --checkpoint /path/to/pi05 --warmup 5 --iters 30
    python dev/pi05_bench.py --arms eager,eager+graph,fused+graph --cameras 2
    python dev/pi05_bench.py --fp8 --steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ENV_FILE = os.path.join(PROJECT_ROOT, "tests", "checkpoints.env")
ROW_PREFIX = "TYBOK_BENCH_ROW "

KNOWN_ARMS = ("eager", "eager+graph", "fused+graph")
DEFAULT_ARMS = "eager+graph,fused+graph"

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


def run_arm(arm: str, args: argparse.Namespace) -> dict:
    """Child side: build one arm's engine, time ``--iters`` requests, return its row."""
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

    frame = engine._profile_frame(args.cameras, args.frame_seed)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        engine.predict_action_chunk(frame)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        engine.predict_action_chunk(frame)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)  # ms

    return {
        "arm": arm,
        "graph": int(graph),
        "fused": int(fused),
        "setup_s": setup_s,
        "iters": len(times),
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[max(0, int(len(times) * 0.95) - 1)],
        "min_ms": min(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "hz": 1000.0 / statistics.mean(times),
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
        default=DEFAULT_ARMS,
        help=f"comma-separated arms ({'|'.join(KNOWN_ARMS)}; default: {DEFAULT_ARMS})",
    )
    parser.add_argument("--warmup", type=int, default=5, help="untimed requests before measuring")
    parser.add_argument("--iters", type=int, default=30, help="timed requests per arm")
    parser.add_argument("--cameras", type=int, default=None, help="cameras in the benchmark frame (default: all)")
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
    parser.add_argument("--fp8", action="store_true", help="add the W8A8 fp8 MLP tiers to fused arms")
    parser.add_argument("--frame-seed", type=int, default=1, help="seed of the synthetic benchmark frame")
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
    unknown = [arm for arm in arms if arm not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {KNOWN_ARMS}")

    if not args.device.startswith("cuda"):
        raise SystemExit("pi05 arms need CUDA (the graph path and the fused tiers are CUDA-only)")

    import torch

    print(f"torch {torch.__version__} | device {args.device} | checkpoint {args.checkpoint}")
    print(
        f"frame seed {args.frame_seed} | cameras {args.cameras or 'all'} | steps {args.steps or 'config'} "
        f"| warmup {args.warmup} | iters {args.iters}"
    )
    print(f"gpu {torch.cuda.get_device_name(0)}")

    argv = list(sys.argv[1:])
    rows = {}
    for arm in arms:
        print(f"[bench] measuring arm '{arm}' ...", flush=True)
        rows[arm] = spawn(arm, argv)
        row = rows[arm]
        if "error" in row:
            print(f"  FAILED: {row['error']}")
        else:
            print(
                f"  mean {row['mean_ms']:.2f} ms | p50 {row['p50_ms']:.2f} | p95 {row['p95_ms']:.2f} "
                f"| {row['hz']:.1f} Hz (setup {row['setup_s']:.1f}s)"
            )

    print("\n=== pi05: predict_action_chunk latency ===")
    print(
        f"{'arm':<14} {'graph':>5} {'fused':>5} {'setup(s)':>8} {'mean(ms)':>9} {'p50(ms)':>9} "
        f"{'p95(ms)':>9} {'min(ms)':>9} {'std(ms)':>8} {'Hz':>7}"
    )
    for arm in arms:
        row = rows[arm]
        if "error" in row:
            print(f"{arm:<14} {'-':>5} {'-':>5} {'-':>8} {'-':>9} {'-':>9} {'-':>9} {'-':>9} {'-':>8} {'-':>7}")
            continue
        print(
            f"{arm:<14} {row['graph']:>5} {row['fused']:>5} {row['setup_s']:>8.1f} {row['mean_ms']:>9.2f} "
            f"{row['p50_ms']:>9.2f} {row['p95_ms']:>9.2f} {row['min_ms']:>9.2f} {row['std_ms']:>8.2f} {row['hz']:>7.1f}"
        )

    failed = [arm for arm in arms if "error" in rows[arm]]
    if "eager+graph" in rows and "fused+graph" in rows and not failed:
        ratio = rows["eager+graph"]["mean_ms"] / rows["fused+graph"]["mean_ms"]
        if ratio >= 1.0:
            print(f"\nfused+graph speedup over eager+graph: {ratio:.2f}x")
        else:
            print(f"\nfused+graph slowdown vs eager+graph: {1.0 / ratio:.2f}x")
    if failed:
        print(f"\nRESULT: FAIL ({', '.join(failed)})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
