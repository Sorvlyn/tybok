#!/usr/bin/env python3
"""Minimal in-process inference script — no gateway / worker / WebSocket.

Loads the engine, runs one inference and prints the result: convenient for debugging
(breakpoints, logs, profiling) and for a quick sanity check of a checkpoint without
starting the server stack.

Usage (run from the project root, i.e. the ``TyBoK/`` directory)::

    # synthetic frame (zero images + given state/task)
    python examples/run_inference.py --checkpoint <ckpt> \\
        --state 0,0,0,0,0,0

    # replay a reference frame dump; --noise-zero makes the result deterministic
    # and comparable to that dump (chunk[0] should match its chunk_post[0])
    python examples/run_inference.py --checkpoint ... --frame <frame.pt> \
        --noise-zero --chunk

    # action-queue semantics: several select_action calls before a re-inference
    python examples/run_inference.py --checkpoint ... --frame <frame.pt> \
        --noise-zero --steps 5

    # engine configuration (same flags as `python -m tybok serve/worker`)
    python examples/run_inference.py --checkpoint ... --graph --graph-cameras 2,3 --noise-zero --chunk
    python examples/run_inference.py --checkpoint ... --tl-fused-expert \
        --noise-zero --chunk
    python examples/run_inference.py --checkpoint ... --graph --seed 42 --denoise-steps 8 --chunk
    python examples/run_inference.py --checkpoint ... --profile --cameras 2   # latency report at load
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from tybok.registry import create_engine  # noqa: E402


def load_reference_frame(path: str):
    """Unpack a ``frame.pt`` dump into (images, state, task)."""
    frame = torch.load(path, weights_only=False)
    images = {}
    for key, val in frame.items():
        if key.startswith("observation.images."):
            images[key[len("observation.images.") :]] = val
    state = frame["observation.state"]
    task = frame.get("task", "")
    return images, state, task


def _build_parser() -> argparse.ArgumentParser:
    """The script's command line: checkpoint / frame selection, then the engine flags.

    The engine flags mirror ``python -m tybok serve/worker --help``; the module docstring lists
    ready-made combinations.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to the (merged) policy checkpoint")
    parser.add_argument("--model", default=None, help="model type (default: auto-detect from config.json)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame", default=None, help="replay a frame.pt dump instead of a synthetic frame")
    parser.add_argument("--task", default="pick up the cup", help="task text (synthetic frame only)")
    parser.add_argument("--state", default="", help="comma-separated floats (synthetic frame only)")
    parser.add_argument("--steps", type=int, default=1, help="number of select_action calls")
    parser.add_argument("--chunk", action="store_true", help="predict the full chunk instead of a single action")
    parser.add_argument("--noise-zero", action="store_true", help="deterministic zero noise (compare with reference)")
    parser.add_argument("--no-warmup", action="store_true")
    # Engine configuration (mirrors `python -m tybok serve/worker --help`).
    parser.add_argument("--compile", action="store_true", help="torch.compile (fast, changes numerics; ~2min build)")
    parser.add_argument("--graph", action="store_true", help="CUDA graphs (bit-exact; needs CUDA)")
    parser.add_argument(
        "--graph-cameras", default=None, help="comma-separated camera counts to pre-capture, e.g. '2,3'"
    )

    parser.add_argument("--tl-fused-vit", action="store_true", help="pi05: fused vision (ViT) kernels (drift tier)")
    parser.add_argument("--tl-llm-flash-attn", action="store_true", help="pi05: fused LLM prefill kernels (drift tier)")
    parser.add_argument(
        "--tl-llm-fused-attn", action="store_true", help="smolvla: fused LLM prefill kernels (drift tier)"
    )
    parser.add_argument(
        "--tl-vit-oproj", action="store_true", help="smolvla: fused vision out_proj kernel (drift tier)"
    )
    parser.add_argument(
        "--tl-fused-expert", action="store_true", help="fused denoising expert attention + MLP (drift tier)"
    )
    parser.add_argument("--tl-fp8-llm-mlp", action="store_true", help="fp8 W8A8 VLM prefill MLP (drift tier)")
    parser.add_argument("--tl-fp8-expert-mlp", action="store_true", help="fp8 W8A8 denoising expert MLP (drift tier)")
    parser.add_argument("--denoise-steps", type=int, default=None, help="Euler/Heun step count override")
    parser.add_argument("--seed", type=int, default=None, help="fixed RNG for the denoising noise (reproducible)")
    parser.add_argument("--sampler", default="euler", choices=["euler", "heun"], help="denoising sampler")
    parser.add_argument("--no-overlap", action="store_true", help="disable the prefill x step-0 graph pipeline")
    parser.add_argument(
        "--no-expert-prefix-kv-cache",
        action="store_true",
        help="disable the expert prefix-KV projection cache (smolvla only; bit-exact either way)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="print a one-shot latency report (wall + phases) after load"
    )
    # Real-Time Chunking (pi05; no extra weights)
    parser.add_argument("--rtc", action="store_true", help="enable RTC guidance (default schedule/horizon)")
    parser.add_argument(
        "--rtc-schedule",
        default="linear",
        choices=["linear", "zeros", "ones", "exp"],
        help="RTC prefix-attention schedule (default: linear)",
    )
    parser.add_argument("--rtc-horizon", type=int, default=10, help="RTC execution horizon (default: 10)")
    parser.add_argument(
        "--rtc-prev",
        default=None,
        help="prev_chunk_left_over as JSON, e.g. '[[0.1,0.2,...],[0.3,0.4,...]]' "
        "(model-space normalized actions; enables guidance on the chunk)",
    )
    parser.add_argument("--rtc-delay", type=int, default=0, help="inference_delay in steps (default: 0)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    graph_cameras = None
    if args.graph_cameras:
        graph_cameras = tuple(int(x) for x in args.graph_cameras.split(",") if x.strip())

    t0 = time.perf_counter()
    rtc_config = None
    if args.rtc:
        rtc_config = {
            "enabled": True,
            "prefix_attention_schedule": args.rtc_schedule.upper(),
            "execution_horizon": args.rtc_horizon,
        }
    engine = create_engine(
        args.checkpoint,
        model_type=args.model,
        device=args.device,
        compile_model=args.compile,
        graph=args.graph,
        graph_cameras=graph_cameras,
        tl_fused_vit=args.tl_fused_vit,
        tl_llm_flash_attn=args.tl_llm_flash_attn,
        tl_llm_fused_attn=args.tl_llm_fused_attn,
        tl_vit_oproj=args.tl_vit_oproj,
        tl_fused_expert=args.tl_fused_expert,
        tl_fp8_llm_mlp=args.tl_fp8_llm_mlp,
        tl_fp8_expert_mlp=args.tl_fp8_expert_mlp,
        num_steps=args.denoise_steps,
        seed=args.seed,
        sampler=args.sampler,
        cache_expert_prefix_kv=not args.no_expert_prefix_kv_cache,
        overlap=not args.no_overlap,
        profile=args.profile,
        warmup=not args.no_warmup,
        rtc_config=rtc_config,
    )
    path = "cuda-graph" if engine.graph_enabled else "compile" if engine.compile_model else "eager"
    drift = (
        "+".join(
            name
            for name, on in (
                ("tl-fused-vit", getattr(engine, "tl_fused_vit", False)),
                ("tl-llm-flash-attn", getattr(engine, "tl_llm_flash_attn", False)),
                ("tl-llm-fused-attn", getattr(engine, "tl_llm_fused_attn", False)),
                ("tl-vit-oproj", getattr(engine, "tl_vit_oproj", False)),
                ("tl-fused-expert", getattr(engine, "tl_fused_expert", False)),
                ("tl-fp8-llm-mlp", getattr(engine, "tl_fp8_llm_mlp", False)),
                ("tl-fp8-expert-mlp", getattr(engine, "tl_fp8_expert_mlp", False)),
            )
            if on
        )
        or "bit-exact"
    )
    print(
        f"[run] engine ready in {time.perf_counter() - t0:.1f}s | path={path} {drift} "
        f"steps={engine.config.num_steps} sampler={engine.config.sampler} | spec={engine.describe()}"
    )

    if args.frame:
        images, state, task = load_reference_frame(args.frame)
        print(f"[run] replayed frame {args.frame}: cameras={list(images)} state={tuple(state.shape)}")
    else:
        spec = engine.describe()
        w, h = spec["resize"]
        images = {cam: torch.zeros(3, h, w, dtype=torch.float32) for cam in spec["cameras"]}
        state = [float(x) for x in args.state.split(",") if x != ""] or [0.0] * spec["action_dim"]
        task = args.task
        print(f"[run] synthetic frame: cameras={list(images)} state={state}")

    frame = engine.make_frame(images, state, task)

    noise = None
    if args.noise_zero:
        spec = engine.describe()
        noise = torch.zeros(1, spec["chunk_size"], spec["action_dim"], dtype=torch.float32)

    t1 = time.perf_counter()
    if args.chunk:
        rtc_kwargs = {}
        if args.rtc and args.rtc_prev:
            import json

            rtc_kwargs["prev_chunk_left_over"] = json.loads(args.rtc_prev)
            rtc_kwargs["inference_delay"] = args.rtc_delay
        if rtc_kwargs:
            action = engine.predict_action_chunk(frame, noise=noise, **rtc_kwargs)
            print(f"[run] chunk (RTC-guided) {tuple(action.shape)}: first step {action[0].tolist()}")
        else:
            action = engine.predict_action_chunk(frame, noise=noise)
            print(f"[run] chunk {tuple(action.shape)}: first step {action[0].tolist()}")
    else:
        for i in range(args.steps):
            action = engine.select_action(frame, noise=noise)
            print(f"[run] action #{i}: {action.tolist()}")
    if engine.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"[run] inference took {time.perf_counter() - t1:.2f}s")


if __name__ == "__main__":
    main()
