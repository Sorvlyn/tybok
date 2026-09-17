"""CLI entry point: ``python -m tybok <command>``.

Commands:

- ``worker``   : start the inference worker (IPC server, owns the model)
- ``gateway``  : start the WebSocket gateway (image processing + WS endpoint)
- ``serve``    : start worker + gateway together (single process tree)
- ``validate`` : compare a model backend against its reference dumps
- ``models``   : list the shipped model backends

(Server-side only; the example WebSocket client lives in ``examples/client.py``.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from .logging_utils import setup_logging
from .protocol import WORKER_STARTUP_TIMEOUT

log = logging.getLogger("tybok.serve")

# Set on the worker/gateway children spawned by ``tybok serve``: the parent already logged
# the one command the operator ran, so the children must not log their derived command again
# (it would turn one deployment command line into three).
_SERVE_CHILD_ENV = "TYBOK_SERVE_CHILD"

# ``--overlap`` / ``--no-overlap`` and ``--profile`` are defined on the worker / serve /
# validate subparsers (plus a hand-kept copy in ``worker.py``'s own parser); the text lives
# here so the three subparsers in this file cannot drift apart.
_OVERLAP_HELP = (
    "overlap the prefix prefill with the first denoising step (smolvla/pi05: "
    "`prefill_layer` x `step0_layer`; fastwam: `prefill_video_layer` x the step-0 "
    "`action_layer`), so the step-0 layers hide inside the prefill window; `--no-overlap` "
    "runs the two strictly in sequence instead. Bit-exact either way (same kernels, "
    "different launch order). On by default wherever it applies: smolvla/pi05 only on the "
    "--graph path (CUDA + euler), fastwam on eager as well; --compile disables it. The "
    "eager-mode gain is platform-dependent and has not been broadly tested"
)
_PROFILE_HELP = (
    "run a one-shot latency profile at startup and log it through the backend logger: wall "
    "time (predict_action_chunk ms/iter + Hz) and, where the backend implements it, a "
    "per-phase breakdown. Startup and worker-side, so it excludes the gateway/IPC hop; the "
    "gateway's --timing is the complementary per-request runtime counter"
)


def _add_camera_alias_arg(parser: argparse.ArgumentParser, *, gateway_side: bool = False) -> None:
    """Add ``--camera-alias``: serve client camera SRC in checkpoint image slot DST.

    SRC is always the *client* side -- the camera key the robot / client actually sends.
    DST is always the *model* side -- an image key of the checkpoint ``config.json``
    (``observation.images.<slot>``), i.e. one of the slots the worker reports in
    ``describe()``. The rename is only applied when DST is missing from the frame, so an
    exact-named frame is never touched. Worker / serve / validate rename it in the engine
    preprocessor; the gateway renames it before forwarding.
    """
    example = "--camera-alias left_wrist_image=image2,right_wrist_image=image3"
    if gateway_side:
        help_text = (
            "gateway: decode client camera SRC into the checkpoint image slot DST (an image "
            "key of the model config.json, e.g. image2; the slots are the camera names the "
            "worker reports in describe()). Applied only when DST is absent from the payload, "
            "so clients may still send the exact slot names. Several pairs go into one "
            f"comma-separated value: {example}"
        )
    else:
        help_text = (
            "rename client camera SRC to checkpoint image slot DST: the client sends SRC "
            "(e.g. left_wrist_image), the engine serves it in the slot DST that the model "
            "config.json expects (an image key, e.g. image2). Applied only when DST is "
            "absent from the frame. Several pairs go into one comma-separated value: "
            f"{example}"
        )
    parser.add_argument(
        "--camera-alias",
        default=None,
        metavar="SRC=DST[,SRC=DST...]",
        help=help_text,
    )


def _deploy_command(argv: list[str] | None, args: argparse.Namespace) -> str:
    """The command this process was started with, as a copy-pasteable string.

    ``python`` rather than ``sys.executable``: the interpreter's absolute path (a conda
    env prefix) is long and adds nothing to what the deployment log is for. The default-on
    switches the operator left out are appended (see ``_default_on_flags``) so that this one
    line describes the whole deployment instead of only the difference from the defaults.
    """
    raw = sys.argv[1:] if argv is None else list(argv)
    return shlex.join(["python", "-m", "tybok", *raw, *_default_on_flags(argv, args)])


# Boolean switches whose default is *on*: the deploy log spells out the ones a run still has on
# and that the operator did not type. They *are* the defaults, so appending them keeps the
# logged command copy-pasteable and behaviour-identical. ``--overlap`` is common to all three
# backends; the rest are fastwam's precision defaults -- the CLI defines those flags for every
# subcommand, but the smolvla/pi05 engines never see them, so they are logged for fastwam only.
_COMMON_DEFAULT_ON_FLAGS = (("overlap", "--overlap"),)
_BACKEND_DEFAULT_ON_FLAGS = {
    "fastwam": (
        ("video_fp8", "--video-fp8"),
        ("action_fp8", "--action-fp8"),
        ("text_fp8", "--text-fp8"),
        ("text_emb_cpu", "--text-emb-cpu"),
    ),
}


def _served_model_type(args: argparse.Namespace) -> str | None:
    """``--model-type``, else the checkpoint ``config.json`` ``type``; None if unknown.

    Reads the JSON instead of calling ``registry.detect_model_type``: that would import the
    engine stack (torch) into the ``serve`` parent, which only spawns children.
    """
    if getattr(args, "model_type", None):
        return args.model_type
    model = getattr(args, "model", None)
    if not model:
        return None
    try:
        with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
            return json.load(f).get("type")
    except (OSError, ValueError):
        return None  # unknown -> log the backend-agnostic defaults only


def _default_on_flags(argv: list[str] | None, args: argparse.Namespace) -> list[str]:
    """Default-on switches this run keeps but the operator did not type."""
    table = _COMMON_DEFAULT_ON_FLAGS + _BACKEND_DEFAULT_ON_FLAGS.get(_served_model_type(args) or "", ())
    typed = set(sys.argv[1:] if argv is None else argv)
    return [flag for dest, flag in table if getattr(args, dest, False) and flag not in typed]


def _cmd_serve(args: argparse.Namespace) -> None:
    """Spawn the worker and the gateway as two child processes (plan 2 layout)."""
    python = sys.executable
    worker_cmd = [
        python,
        "-m",
        "tybok",
        "worker",
        "--model",
        args.model,
        "--socket",
        args.socket,
        "--device",
        args.device,
    ]
    if args.model_type:
        worker_cmd += ["--model-type", args.model_type]
    if args.compile:
        worker_cmd += ["--compile"]
    if args.graph:
        worker_cmd += ["--graph"]
    if args.graph_cameras:
        worker_cmd += ["--graph-cameras", args.graph_cameras]
    if args.tl_fused_vit:
        worker_cmd += ["--tl-fused-vit"]
    if args.tl_llm_flash_attn:
        worker_cmd += ["--tl-llm-flash-attn"]
    if args.tl_llm_fused_attn:
        worker_cmd += ["--tl-llm-fused-attn"]
    if args.tl_vit_oproj:
        worker_cmd += ["--tl-vit-oproj"]
    if args.tl_fused_expert:
        worker_cmd += ["--tl-fused-expert"]
    if args.tl_fp8_llm_mlp:
        worker_cmd += ["--tl-fp8-llm-mlp"]
    if args.tl_fp8_expert_mlp:
        worker_cmd += ["--tl-fp8-expert-mlp"]
    if args.vit_mlp_dtype:
        worker_cmd += ["--vit-mlp-dtype", args.vit_mlp_dtype]
    if args.skip_empty_cams:
        worker_cmd += ["--skip-empty-cams"]
    if args.pad_free:
        worker_cmd += ["--pad-free"]
    if getattr(args, "pack_qkv", False):
        worker_cmd += ["--pack-qkv"]
    if not getattr(args, "video_fp8", True):
        worker_cmd += ["--video-bf16"]
    if not getattr(args, "action_fp8", True):
        worker_cmd += ["--action-bf16"]
    if getattr(args, "cu_fused_vdit", False):
        worker_cmd += ["--cu-fused-vdit"]
    if getattr(args, "cu_fused_adit", False):
        worker_cmd += ["--cu-fused-adit"]
    if getattr(args, "cu_fused_text_encoder", False):
        worker_cmd += ["--cu-fused-text-encoder"]
    if getattr(args, "cooperative_kernel", False) and (
        getattr(args, "cu_fused_vdit", False)
        or getattr(args, "cu_fused_adit", False)
        or getattr(args, "cu_fused_text_encoder", False)
    ):
        worker_cmd += ["--cooperative-kernel"]
    if not getattr(args, "text_fp8", True):
        worker_cmd += ["--text-bf16"]
    if getattr(args, "no_prompt_cache", False):
        worker_cmd += ["--no-prompt-cache"]
    if getattr(args, "no_action_context_cache", False):
        worker_cmd += ["--no-action-context-cache"]
    if getattr(args, "action_pre_fused", False):
        worker_cmd += ["--action-pre-fused"]
    if getattr(args, "video_pre_fused", False):
        worker_cmd += ["--video-pre-fused"]
    if getattr(args, "require_fused", False):
        worker_cmd += ["--require-fused"]
    if args.camera_alias:
        worker_cmd += ["--camera-alias", args.camera_alias]
    if args.steps:
        worker_cmd += ["--steps", str(args.steps)]
    if args.seed is not None:
        worker_cmd += ["--seed", str(args.seed)]
    if args.sampler != "euler":
        worker_cmd += ["--sampler", args.sampler]
    if args.no_expert_prefix_kv_cache:
        worker_cmd += ["--no-expert-prefix-kv-cache"]
    if not args.overlap:
        worker_cmd += ["--no-overlap"]
    if args.tokenizer_dir:
        worker_cmd += ["--tokenizer-dir", args.tokenizer_dir]
    if getattr(args, "text_encoder_dir", None):
        worker_cmd += ["--text-encoder-dir", args.text_encoder_dir]
    if getattr(args, "vae_dir", None):
        worker_cmd += ["--vae-dir", args.vae_dir]
    if getattr(args, "text_encoder_device", None) not in (None, "cpu"):
        worker_cmd += ["--text-encoder-device", args.text_encoder_device]
    if not getattr(args, "text_emb_cpu", True):
        worker_cmd += ["--no-text-emb-cpu"]
    if args.profile:
        worker_cmd += ["--profile"]
    if args.shm_ipc:
        worker_cmd += ["--shm-ipc"]
    if args.rtc:
        worker_cmd += ["--rtc"]
        worker_cmd += ["--rtc-schedule", args.rtc_schedule]
        worker_cmd += ["--rtc-max-guidance-weight", str(args.rtc_max_guidance_weight)]
        worker_cmd += ["--rtc-execution-horizon", str(args.rtc_execution_horizon)]
        if args.rtc_debug:
            worker_cmd += ["--rtc-debug"]
        if args.rtc_debug_maxlen != 100:
            worker_cmd += ["--rtc-debug-maxlen", str(args.rtc_debug_maxlen)]
    gateway_cmd = [
        python,
        "-m",
        "tybok",
        "gateway",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--worker-socket",
        args.socket,
    ]
    if args.ipc_uint8:
        gateway_cmd += ["--ipc-uint8"]
    if args.shm_ipc:
        gateway_cmd += ["--shm-ipc"]
    if args.gpu_ipc:
        if args.shm_ipc:
            raise RuntimeError("--shm-ipc and --gpu-ipc are mutually exclusive")
        gateway_cmd += ["--gpu-ipc", "--gpu-slots", str(args.gpu_slots)]
    if args.max_inflight:
        gateway_cmd += ["--max-inflight", str(args.max_inflight)]
    if args.camera_alias:
        gateway_cmd += ["--camera-alias", args.camera_alias]

    procs = []

    # tear down children when this process receives SIGTERM (e.g. from an
    # orchestrator or test harness)
    def _on_term(*_):
        _shutdown(procs, args.socket)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_term)
    # the children get the marker so they stay quiet about their derived commands
    child_env = {**os.environ, _SERVE_CHILD_ENV: "1"}
    try:
        procs.append(subprocess.Popen(worker_cmd, env=child_env))
        log.info(f"worker process pid={procs[0].pid}")
        # wait for the worker socket to appear: the worker creates it once the model is
        # loaded (and any graph captured), so its presence means "worker ready". The budget
        # is ``protocol.WORKER_STARTUP_TIMEOUT`` (an old/slow CPU can take minutes); the
        # gateway waits for the same budget on its own, so starting it early is safe too.
        deadline = time.time() + WORKER_STARTUP_TIMEOUT
        while time.time() < deadline:
            if os.path.exists(args.socket):
                break
            if procs[0].poll() is not None:
                raise RuntimeError("worker exited early; see its stderr")
            time.sleep(0.5)
        procs.append(subprocess.Popen(gateway_cmd, env=child_env))
        log.info(f"gateway process pid={procs[1].pid} ws://{args.host}:{args.port}/ws")
        log.info(
            f"deployment complete: model={args.model} host={args.host} port={args.port} "
            f"ws=ws://{args.host}:{args.port}/ws health=http://{args.host}:{args.port}/health "
            f"worker_socket={args.socket}"
        )
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        _shutdown(procs, args.socket)


def _shutdown(procs: list, socket_path: str) -> None:
    log.info("shutting down ...")
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    if socket_path and os.path.exists(socket_path):
        os.unlink(socket_path)
    if socket_path and os.path.exists(f"{socket_path}.shm"):
        os.unlink(f"{socket_path}.shm")


def _cmd_models(args: argparse.Namespace) -> None:
    from .registry import shipped_models

    # ``shipped_models`` rather than ``available``: this listing is the dependency-free part of
    # the CLI, and importing the backends to enumerate them would drag numpy / torch in.
    print("shipped model backends:")
    for name in shipped_models():
        print(f"  - {name}")


def _cmd_validate(args: argparse.Namespace) -> None:
    from .registry import create_engine
    from .worker import parse_camera_alias

    camera_alias = parse_camera_alias(args.camera_alias)
    engine = create_engine(
        args.model,
        model_type=args.model_type,
        device=args.device,
        compile_model=args.compile,
        graph=args.graph,
        tl_fused_vit=args.tl_fused_vit,
        tl_llm_flash_attn=args.tl_llm_flash_attn,
        tl_llm_fused_attn=args.tl_llm_fused_attn,
        tl_vit_oproj=args.tl_vit_oproj,
        tl_fused_expert=args.tl_fused_expert,
        tl_fp8_llm_mlp=args.tl_fp8_llm_mlp,
        tl_fp8_expert_mlp=args.tl_fp8_expert_mlp,
        vit_mlp_dtype=args.vit_mlp_dtype,
        skip_empty_images=args.skip_empty_cams,
        pad_free=args.pad_free,
        pack_qkv=args.pack_qkv,
        **({"video_fp8": False} if not getattr(args, "video_fp8", True) else {}),
        **({"action_fp8": False} if not getattr(args, "action_fp8", True) else {}),
        **({"text_fp8": False} if not getattr(args, "text_fp8", True) else {}),
        action_context_cache=not getattr(args, "no_action_context_cache", False),
        text_fused=getattr(args, "cu_fused_text_encoder", False),
        text_fused_split=(
            getattr(args, "cu_fused_text_encoder", False) and not getattr(args, "cooperative_kernel", False)
        ),
        action_fused=getattr(args, "cu_fused_adit", False),
        action_fused_split=(getattr(args, "cu_fused_adit", False) and not getattr(args, "cooperative_kernel", False)),
        require_fused=getattr(args, "require_fused", False),
        num_steps=args.steps,
        sampler=args.sampler,
        cache_expert_prefix_kv=not args.no_expert_prefix_kv_cache,
        cache_prompt=not args.no_prompt_cache,
        overlap=args.overlap,
        action_pre_fused=getattr(args, "action_pre_fused", False),
        video_pre_fused=getattr(args, "video_pre_fused", False),
        **({"camera_alias": camera_alias} if camera_alias else {}),
        **({"tokenizer_dir": args.tokenizer_dir} if args.tokenizer_dir else {}),
        **({"text_encoder_dir": args.text_encoder_dir} if args.text_encoder_dir else {}),
        **({"vae_dir": args.vae_dir} if args.vae_dir else {}),
        **({"text_encoder_device": args.text_encoder_device} if getattr(args, "text_encoder_device", None) else {}),
        **({"text_emb_cpu": False} if not getattr(args, "text_emb_cpu", True) else {}),
    )
    if not args.reference:
        raise SystemExit("validate needs the reference dump directory: pass --reference DIR or set TYBOK_REFERENCE_DIR")
    errors = engine.validate(args.reference)
    # the validation report printer lives next to each engine implementation
    import importlib

    report_mod = importlib.import_module(type(engine).__module__)
    report_mod.print_validation_report(errors)
    fast_like = (
        args.tl_fused_vit
        or args.tl_llm_flash_attn
        or args.tl_llm_fused_attn
        or args.tl_vit_oproj
        or args.tl_fused_expert
        or args.tl_fp8_llm_mlp
        or args.tl_fp8_expert_mlp
        or args.vit_mlp_dtype
        or args.pad_free
        or args.pack_qkv
        or getattr(args, "action_pre_fused", False)
        or args.steps is not None
    )
    if fast_like:
        # the drift tiers / --tl-* / --steps gate on the delivered action chunk: the
        # internal prefix KV has a large absolute scale that is not a quality
        # metric.
        gate = errors.get("chunk_post.max", 0.0)
        label = "chunk_post.max"
    else:
        gate = max(v for k, v in errors.items() if k.endswith(".max"))
        label = "worst component"
    tol = args.tolerance
    if tol is None:
        tol = 0.05 if fast_like else 1e-3
    if gate == 0.0:
        print("PASS: engine matches the reference implementation exactly.")
    elif gate < tol:
        print(f"PASS: {label} error {gate:.4g} within tolerance {tol:g}.")
    else:
        print(f"FAIL: {label} error {gate:.4g} exceeds tolerance {tol:g}.")


def _add_fastwam_args(parser: argparse.ArgumentParser) -> None:
    """Shared fastWAM sidecar flags (worker / serve / validate subparsers).

    The UMT5 text encoder, the Wan2.2 VAE and the tokenizer (``--tokenizer-dir``, defined
    on the shared parser above) are not part of the fastwam checkpoint; these flags point
    the deployment at local copies of them.
    """
    parser.add_argument(
        "--text-encoder-dir",
        default=None,
        help="fastwam: UMT5 text-encoder weights dir (default: the checkpoint config's "
        "text_encoder_model_id resolved locally; expects model.safetensors (fp8 or "
        "bf16 shards) + config.json)",
    )
    parser.add_argument(
        "--vae-dir",
        default=None,
        help="fastwam: Wan2.2 VAE dir (diffusers AutoencoderKLWan weights + config.json; "
        "default: sibling `vae/` of the text-encoder dir)",
    )
    parser.add_argument(
        "--text-encoder-device",
        default=None,
        help="fastwam: device for the UMT5 text encoder (default: cpu, keeping VRAM for "
        "the ~12GB DiT; use `cuda` to keep UMT5 fp8-resident on GPU with the embedding "
        "on CPU)",
    )
    parser.add_argument(
        "--text-emb-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fastwam: keep the text-encoder *embedding table* (2GB bf16, a pure lookup, no "
        "matmul) on CPU while the rest runs on GPU (default). Use --no-text-emb-cpu to move "
        "the table to GPU too (only relevant with --text-encoder-device cuda).",
    )
    parser.add_argument(
        "--pack-qkv",
        action="store_true",
        help="fastwam: pack each DiT attention's q/k/v into one GEMM (self 3x / cross 2x); "
        "bigger GEMMs are fp8-friendlier and cut action-expert launch count",
    )
    parser.add_argument(
        "--video-fp8",
        dest="video_fp8",
        action="store_true",
        default=True,
        help="fastwam: run the video expert fp8-resident W8A8 (default; loads "
        "model.fp8.safetensors when present, saving ~5GB VRAM); video DiT only",
    )
    parser.add_argument(
        "--video-bf16",
        dest="video_fp8",
        action="store_false",
        help="fastwam: run the video expert in bf16 instead of fp8 (video DiT only)",
    )
    parser.add_argument(
        "--action-fp8",
        dest="action_fp8",
        action="store_true",
        default=True,
        help="fastwam: run the action expert fp8-resident W8A8 (default; saving ~1GB VRAM); action expert only",
    )
    parser.add_argument(
        "--action-bf16",
        dest="action_fp8",
        action="store_false",
        help="fastwam: run the action expert in bf16 instead of fp8 (action expert only)",
    )
    parser.add_argument(
        "--cu-fused-adit",
        action="store_true",
        default=False,
        help="fastwam: run the action denoise layers with fused CUDA kernels (needs an "
        "fp8-resident action expert and a CUDA device); with --cooperative-kernel uses "
        "cooperative launches",
    )
    parser.add_argument(
        "--cu-fused-vdit",
        action="store_true",
        default=False,
        help="fastwam: run the video prefill blocks with fused CUDA kernels (needs an "
        "fp8-resident video expert and a CUDA device); with --cooperative-kernel uses "
        "cooperative launches",
    )
    parser.add_argument(
        "--cooperative-kernel",
        action="store_true",
        default=False,
        help="fastwam: make the enabled --cu-fused-* families use cooperative launches. "
        "The grid is sized from the *current* device (occupancy x SM count), so a bigger "
        "card is not reserved whole; the real constraint is that the whole grid must be "
        "resident at once, so other workloads (or MPS partitions) sharing the GPU can fail "
        "the launch. Falls back to the split form unless --require-fused",
    )
    parser.add_argument(
        "--require-fused",
        action="store_true",
        default=False,
        help="fastwam: error out instead of degrading when a fused path cannot run "
        "(fused cooperative -> fused split -> eager)",
    )
    parser.add_argument(
        "--text-fp8",
        dest="text_fp8",
        action="store_true",
        default=True,
        help="fastwam: run the text encoder fp8-resident W8A8 (default; needs "
        "--text-encoder-device cuda and fp8 files in the dir)",
    )
    parser.add_argument(
        "--text-bf16",
        dest="text_fp8",
        action="store_false",
        help="fastwam: run the text encoder in bf16 instead of fp8",
    )
    parser.add_argument(
        "--no-prompt-cache",
        action="store_true",
        help="fastwam: re-encode the task prompt (UMT5) on every request instead of reusing "
        "the single-entry prompt memo (bit-identical; for profiling / benchmarking, where "
        "the cached path would hide the text-encoder cost).",
    )
    parser.add_argument(
        "--no-action-context-cache",
        action="store_true",
        help="fastwam: recompute the action expert cross-attention context (text embedding + "
        "per-layer k/v) every denoising step instead of once per chunk (bit-exact either "
        "way; A/B timing knob).",
    )
    parser.add_argument(
        "--cu-fused-text-encoder",
        action="store_true",
        help="fastwam: run the two UMT5 sublayers (self-attention + FFN) with fused CUDA "
        "kernels (needs --text-encoder-device cuda + fp8 + a CUDA device); with "
        "--cooperative-kernel uses cooperative launches",
    )
    parser.add_argument(
        "--action-pre-fused",
        action="store_true",
        default=False,
        help="fastwam: fuse the action DiT pre-components (time path + context k/v "
        "precompute); requires --cu-fused-adit",
    )
    parser.add_argument(
        "--video-pre-fused",
        action="store_true",
        default=False,
        help="fastwam: exact video pre-DiT optimizations (resident RoPE tables, prebuilt "
        "frequency table, cached mask/grid)",
    )


def _add_rtc_args(parser: argparse.ArgumentParser) -> None:
    """Shared Real-Time Chunking flags (worker / serve subparsers)."""
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="enable Real-Time Chunking guidance (pi0.5; no extra weights). The client "
        "then drives RTC per request (prev_chunk_left_over / inference_delay / "
        "execution_horizon) and receives the normalized chunk alongside the action",
    )
    parser.add_argument(
        "--rtc-schedule",
        default="linear",
        choices=["linear", "zeros", "ones", "exp"],
        help="RTC prefix-attention weight schedule (default: linear)",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
        help="RTC guidance weight clamp (default: 10.0)",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
        help="RTC execution horizon in steps (default: 10); can be overridden per request",
    )
    parser.add_argument(
        "--rtc-debug",
        action="store_true",
        help="record per-step RTC guidance debug info (Tracker; default: off)",
    )
    parser.add_argument(
        "--rtc-debug-maxlen",
        type=int,
        default=100,
        help="RTC debug tracker sliding window (default: 100)",
    )


# --------------------------------------------------------------------------- #
# CLI
#
# The options live in one place as data (``_Flag``) instead of being pasted into every
# subparser: worker / serve / gateway / validate share most of them, and a copy that drifts is
# how a flag ends up meaning two things. ``_COMMANDS`` then composes each subcommand from those
# definitions, in the order argparse prints them, and names the handler ``main`` dispatches to.
# --------------------------------------------------------------------------- #
_UNSET = object()


@dataclass(frozen=True)
class _Flag:
    """One command-line option, defined once and attached to any number of subcommands.

    ``flags`` is the spelling (e.g. ``("--graph",)``); every other field maps 1:1 onto an
    ``add_argument`` keyword -- ``as_type`` is ``type``, and ``default`` defaults to
    ``_UNSET`` because ``default=None`` is itself a meaningful default for some flags.
    Instances are callables taking a parser, so a shared multi-flag group (see
    ``_add_fastwam_args`` / ``_add_rtc_args``) can be used in the same tuple.
    """

    flags: tuple[str, ...]
    help: str | None = None
    default: object = _UNSET
    action: object = None
    as_type: object = None
    choices: Sequence[str] | None = None
    required: bool = False

    def __call__(self, parser: argparse.ArgumentParser) -> None:
        kwargs: dict[str, Any] = {}
        if self.help is not None:
            kwargs["help"] = self.help
        if self.default is not _UNSET:
            kwargs["default"] = self.default
        if self.action is not None:
            kwargs["action"] = self.action
        if self.as_type is not None:
            kwargs["type"] = self.as_type
        if self.choices is not None:
            kwargs["choices"] = list(self.choices)
        if self.required:
            kwargs["required"] = True
        parser.add_argument(*self.flags, **kwargs)


@dataclass(frozen=True)
class _Command:
    """One subcommand: its top-level help line, its options in ``--help`` order, its handler."""

    help: str
    options: tuple
    handler: Callable[[argparse.Namespace], None]


_MODEL = _Flag(("--model",), required=True, help="path to the model checkpoint dir")

_MODEL_TYPE = _Flag(
    ("--model-type",),
    default=None,
    help="backend override (smolvla/pi05/fastwam); default: the checkpoint config.json 'type'",
)

_SOCKET = _Flag(("--socket",), default="/tmp/tybok_worker.sock")

_DEVICE = _Flag(("--device",), default="auto")

_COMPILE = _Flag(("--compile",), action="store_true")

_GRAPH = _Flag(("--graph",), action="store_true")

_GRAPH_CAMERAS = _Flag(
    ("--graph-cameras",),
    default=None,
    help="comma-separated camera counts whose CUDA-graph shape to pre-capture at startup (e.g. '2,3'); "
    "each camera is its own single-image ViT forward, so this selects prefix-length buckets, not a "
    "ViT batch size",
)

_TL_FUSED_VIT = _Flag(
    ("--tl-fused-vit",),
    action="store_true",
    help="pi0.5: fuse the ViT attention, MLP and projector into Triton kernels",
)

_TL_LLM_FLASH_ATTN = _Flag(
    ("--tl-llm-flash-attn",),
    action="store_true",
    help="pi0.5: VLM prefill attention as a Triton GQA flash kernel (q/k/v concatenated into one GEMM)",
)

_TL_LLM_FUSED_ATTN = _Flag(
    ("--tl-llm-fused-attn",),
    action="store_true",
    help="smolvla: fully fuse the LLM prefill attention in Triton (RMSNorm + q/k/v + RoPE + KV-cache "
    "write + GQA flash)",
)

_TL_VIT_OPROJ = _Flag(
    ("--tl-vit-oproj",),
    action="store_true",
    help="smolvla: vision out_proj as one Triton kernel; q/k/v are concatenated into one GEMM (vision "
    "MLP is not fused)",
)

_TL_FUSED_EXPERT = _Flag(
    ("--tl-fused-expert",),
    action="store_true",
    help="pi0.5 / smolvla: fuse the denoising-expert attention and MLP into Triton kernels",
)

_VIT_MLP_DTYPE = _Flag(
    ("--vit-mlp-dtype",),
    default=None,
    choices=["fp16", "bf16"],
    help="pi0.5: dtype of the ViT MLP GEMMs (fc1/fc2); other ViT ops stay fp32; ignored under --tl-fused-vit",
)

_SKIP_EMPTY_CAMS_WORKER = _Flag(
    ("--skip-empty-cams",),
    action="store_true",
    help="pi0.5: skip the ViT forward for the empty camera slots and pad the prefix with zeros instead "
    "(bit-exact; the check sits in embed_prefix, before the ViT, so it also applies under "
    "--tl-fused-vit). Not available on the --graph path, which bakes all-True masks at capture -- "
    "use --pad-free there",
)

_PAD_FREE = _Flag(
    ("--pad-free",),
    action="store_true",
    help="pi0.5: padding-free VLM prefill (drop empty-camera and language-pad tokens from the prefix)",
)

_TL_FP8_LLM_MLP = _Flag(
    ("--tl-fp8-llm-mlp",),
    action="store_true",
    help="pi0.5: run the VLM prefill MLP as a fused Triton W8A8 fp8 GEMM (requires an fp8-capable GPU)",
)

_TL_FP8_EXPERT_MLP = _Flag(
    ("--tl-fp8-expert-mlp",),
    action="store_true",
    help="pi0.5: run the denoising-expert MLP as a fused Triton W8A8 fp8 chain (requires an fp8-capable GPU)",
)

_STEPS = _Flag(
    ("--steps",),
    as_type=int,
    default=None,
    help="Euler denoising step count (default: checkpoint config)",
)

_SEED = _Flag(
    ("--seed",),
    as_type=int,
    default=None,
    help="seed the denoising-noise RNG (reproducible action sequences)",
)

_SAMPLER = _Flag(
    ("--sampler",),
    default="euler",
    choices=["euler", "heun"],
    help="denoising sampler: euler (reference, bit-exact) or heun (2nd-order; N steps = 2N velocity "
    "evals, not bit-exact)",
)

_NO_EXPERT_PREFIX_KV_CACHE = _Flag(
    ("--no-expert-prefix-kv-cache",),
    action="store_true",
    help="smolvla: disable the expert prefix-KV projection cache (recompute the fp32 expert "
    "cross-attention k/v projections every denoising step; bit-exact either way). No-op for pi05 "
    "(its prefix-KV reuse is structural); fastwam rejects it (its video KV cache is unconditional)",
)

_OVERLAP = _Flag(
    ("--overlap",),
    action=argparse.BooleanOptionalAction,
    default=True,
    help=_OVERLAP_HELP,
)

_PROFILE = _Flag(("--profile",), action="store_true", help=_PROFILE_HELP)

_SHM_IPC_WORKER = _Flag(
    ("--shm-ipc",),
    action="store_true",
    help="share the ~6MB tensor payload with the gateway over a shared-memory region (the socket then "
    "carries only the small JSON header; opt-in, byte-identical to the socket path)",
)

_TOKENIZER_DIR = _Flag(
    ("--tokenizer-dir",),
    default=None,
    help="override the tokenizer dir (PaliGemma for pi0.5; UMT5 for fastwam; default: resolved from the checkpoint)",
)

_HOST = _Flag(("--host",), default="0.0.0.0")

_PORT = _Flag(("--port",), as_type=int, default=8765)

_WORKER_SOCKET = _Flag(("--worker-socket",), default="/tmp/tybok_worker.sock")

_WORKER_HOST = _Flag(("--worker-host",), default="127.0.0.1")

_WORKER_PORT = _Flag(("--worker-port",), as_type=int, default=5555)

_IPC_UINT8 = _Flag(
    ("--ipc-uint8",),
    action="store_true",
    help="quantise the resized [0,1] images to uint8 for the gateway<->worker IPC (4x smaller payload; "
    "1/255 pixel rounding -- opt-in, breaks bit-exactness)",
)

_MAX_INFLIGHT = _Flag(
    ("--max-inflight",),
    as_type=int,
    default=4,
    help="per-connection pipelined messages decoded ahead of the worker response (default 4)",
)

_SHM_IPC_GATEWAY = _Flag(
    ("--shm-ipc",),
    action="store_true",
    help="carry the ~6MB tensor payload over a shared-memory region instead of the socket (the socket "
    "then carries only the small JSON header; opt-in, byte-identical)",
)

_GPU_IPC = _Flag(
    ("--gpu-ipc",),
    action="store_true",
    help="GPU-direct IPC: HtoD the frame on the gateway and share the CUDA buffers with the worker via "
    "cudaIpcMemHandle (opt-in, byte-identical; requires CUDA, mutually exclusive with --shm-ipc)",
)

_GPU_SLOTS = _Flag(
    ("--gpu-slots",),
    as_type=int,
    default=8,
    help="gateway CUDA keep-alive ring depth for --gpu-ipc (default 8)",
)

_TIMING = _Flag(
    ("--timing",),
    action="store_true",
    help="log one [timing] line per request (decode wait + total round trip), like the C++ gateway. "
    "Runtime and per-request, so it includes the gateway -> worker hop; the worker-side --profile "
    "is the complementary startup one-shot that breaks the inference itself down by phase",
)

_SKIP_EMPTY_CAMS_SERVE = _Flag(
    ("--skip-empty-cams",),
    action="store_true",
    help="pi0.5: skip the ViT forward for empty camera slots (bit-exact; eager path only)",
)

_REFERENCE = _Flag(
    ("--reference",),
    default=os.environ.get("TYBOK_REFERENCE_DIR"),
    help="directory holding the reference dumps to compare against (default: $TYBOK_REFERENCE_DIR)",
)

_TOLERANCE = _Flag(
    ("--tolerance",),
    as_type=float,
    default=None,
    help="max-abs tolerance vs the reference dumps (default: 1e-3; 0.05 when a --tl-* / low-precision "
    "flag or --steps is set)",
)


# --------------------------------------------------------------------------- #
# Compositions
#
# Each subcommand is a concatenation of these groups, in the order argparse prints them: a group
# is a plain tuple of the definitions above, so composing one costs nothing at parse time, and a
# flag that three commands share is named once here instead of in each of them.
# --------------------------------------------------------------------------- #
_MODEL_ARGS = (_MODEL, _MODEL_TYPE)
_ENDPOINT_ARGS = (_HOST, _PORT)
_WORKER_TARGET_ARGS = (_WORKER_SOCKET, _WORKER_HOST, _WORKER_PORT)
_ENGINE_ARGS = (_DEVICE, _COMPILE, _GRAPH)
# worker / serve pre-capture one CUDA-graph shape per camera count; validate runs a single shape.
_GRAPH_ARGS = _ENGINE_ARGS + (_GRAPH_CAMERAS,)
_FUSED_ARGS = (_TL_FUSED_VIT, _TL_LLM_FLASH_ATTN, _TL_LLM_FUSED_ATTN, _TL_VIT_OPROJ, _TL_FUSED_EXPERT, _VIT_MLP_DTYPE)
_FP8_ARGS = (_PAD_FREE, _TL_FP8_LLM_MLP, _TL_FP8_EXPERT_MLP)
_CAMERA_ARGS = (_add_camera_alias_arg,)
_SAMPLING_ARGS = (_STEPS, _SEED, _SAMPLER)
_LOOP_ARGS = (_NO_EXPERT_PREFIX_KV_CACHE, _OVERLAP)
_IPC_ARGS = (_IPC_UINT8, _MAX_INFLIGHT, _SHM_IPC_GATEWAY, _GPU_IPC, _GPU_SLOTS)
_FASTWAM_ARGS = (_TOKENIZER_DIR, _add_fastwam_args)
_RTC_ARGS = (_add_rtc_args,)


_COMMANDS = {
    "worker": _Command(
        help="start the inference worker",
        options=(
            _MODEL_ARGS
            + (_SOCKET,)
            + _GRAPH_ARGS
            + _FUSED_ARGS
            + (_SKIP_EMPTY_CAMS_WORKER,)
            + _FP8_ARGS
            + _CAMERA_ARGS
            + _SAMPLING_ARGS
            + _LOOP_ARGS
            + (_PROFILE, _SHM_IPC_WORKER)
            + _FASTWAM_ARGS
            + _RTC_ARGS
        ),
        handler=lambda a: _import_worker(a),
    ),
    "gateway": _Command(
        help="start the WebSocket gateway",
        options=(
            _ENDPOINT_ARGS
            + _WORKER_TARGET_ARGS
            + _IPC_ARGS
            + (_TIMING,)
            + (partial(_add_camera_alias_arg, gateway_side=True),)
        ),
        handler=lambda a: _import_gateway(a),
    ),
    "serve": _Command(
        help="start worker + gateway together",
        options=(
            _MODEL_ARGS
            + (_SOCKET,)
            + _ENDPOINT_ARGS
            + _GRAPH_ARGS
            + _FUSED_ARGS
            + (_SKIP_EMPTY_CAMS_SERVE,)
            + _FP8_ARGS
            + _CAMERA_ARGS
            + _SAMPLING_ARGS
            + _LOOP_ARGS
            + (_PROFILE,)
            + _IPC_ARGS
            + _FASTWAM_ARGS
            + _RTC_ARGS
        ),
        handler=_cmd_serve,
    ),
    "validate": _Command(
        help="validate a model backend against reference dumps",
        options=(
            _MODEL_ARGS
            + (_REFERENCE,)
            + _ENGINE_ARGS
            + _FUSED_ARGS
            + (_SKIP_EMPTY_CAMS_SERVE,)
            + _FP8_ARGS
            + _CAMERA_ARGS
            + (_STEPS, _SAMPLER)  # no --seed: validate compares against dumps, not noise
            + _LOOP_ARGS
            + _FASTWAM_ARGS
            + (_TOLERANCE,)
        ),
        handler=_cmd_validate,
    ),
    "models": _Command(
        help="list the shipped model backends",
        options=(),
        handler=_cmd_models,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI from ``_COMMANDS``: one subparser per command, options added in order."""
    parser = argparse.ArgumentParser(prog="tybok", description="Policy deployment engine")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, command in _COMMANDS.items():
        sub_parser = sub.add_parser(name, help=command.help)
        for option in command.options:
            option(sub_parser)
        sub_parser.set_defaults(func=command.handler)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging()
    # One command line per deployment: this is the command the operator ran, logged under
    # the subcommand's own component (``tybok.serve`` / ``tybok.worker`` /
    # ``tybok.gateway``). ``serve`` spawns the worker/gateway from it; those children run
    # with ``_SERVE_CHILD_ENV`` set and log their derived command at DEBUG only, so the
    # derived commands never clutter the INFO startup log.
    if args.command in ("worker", "gateway", "serve"):
        level = logging.DEBUG if os.environ.get(_SERVE_CHILD_ENV) == "1" else logging.INFO
        logging.getLogger(f"tybok.{args.command}").log(level, "[deploy] command: %s", _deploy_command(argv, args))
    args.func(args)


def _import_worker(args: argparse.Namespace) -> None:
    from .worker import main as worker_main

    worker_main(
        [
            "--model",
            args.model,
            "--socket",
            args.socket,
            "--device",
            args.device,
        ]
        + (["--model-type", args.model_type] if args.model_type else [])
        + (["--compile"] if args.compile else [])
        + (["--graph"] if args.graph else [])
        + (["--graph-cameras", args.graph_cameras] if args.graph_cameras else [])
        + (["--tl-fused-vit"] if args.tl_fused_vit else [])
        + (["--tl-llm-flash-attn"] if args.tl_llm_flash_attn else [])
        + (["--tl-llm-fused-attn"] if args.tl_llm_fused_attn else [])
        + (["--tl-vit-oproj"] if args.tl_vit_oproj else [])
        + (["--tl-fused-expert"] if args.tl_fused_expert else [])
        + (["--tl-fp8-llm-mlp"] if args.tl_fp8_llm_mlp else [])
        + (["--tl-fp8-expert-mlp"] if args.tl_fp8_expert_mlp else [])
        + (["--vit-mlp-dtype", args.vit_mlp_dtype] if args.vit_mlp_dtype else [])
        + (["--steps", str(args.steps)] if args.steps else [])
        + (["--seed", str(args.seed)] if args.seed is not None else [])
        + (["--sampler", args.sampler] if args.sampler != "euler" else [])
        + (["--no-expert-prefix-kv-cache"] if args.no_expert_prefix_kv_cache else [])
        + (["--no-overlap"] if not args.overlap else [])
        + (["--profile"] if args.profile else [])
        + (["--shm-ipc"] if args.shm_ipc else [])
        + (["--tokenizer-dir", args.tokenizer_dir] if getattr(args, "tokenizer_dir", None) else [])
        + (["--text-encoder-dir", args.text_encoder_dir] if getattr(args, "text_encoder_dir", None) else [])
        + (["--vae-dir", args.vae_dir] if getattr(args, "vae_dir", None) else [])
        + (["--text-encoder-device", args.text_encoder_device] if getattr(args, "text_encoder_device", None) else [])
        + (["--no-text-emb-cpu"] if not getattr(args, "text_emb_cpu", True) else [])
        + (["--rtc"] if getattr(args, "rtc", False) else [])
        + (
            [
                "--rtc-schedule",
                args.rtc_schedule,
                "--rtc-max-guidance-weight",
                str(args.rtc_max_guidance_weight),
                "--rtc-execution-horizon",
                str(args.rtc_execution_horizon),
            ]
            if getattr(args, "rtc", False)
            else []
        )
        + (["--rtc-debug"] if getattr(args, "rtc_debug", False) else [])
        + (["--skip-empty-cams"] if getattr(args, "skip_empty_cams", False) else [])
        + (["--pad-free"] if getattr(args, "pad_free", False) else [])
        + (["--pack-qkv"] if getattr(args, "pack_qkv", False) else [])
        + (["--video-bf16"] if not getattr(args, "video_fp8", True) else [])
        + (["--action-bf16"] if not getattr(args, "action_fp8", True) else [])
        + (["--cu-fused-vdit"] if getattr(args, "cu_fused_vdit", False) else [])
        + (["--cu-fused-adit"] if getattr(args, "cu_fused_adit", False) else [])
        + (["--cu-fused-text-encoder"] if getattr(args, "cu_fused_text_encoder", False) else [])
        + (
            ["--cooperative-kernel"]
            if getattr(args, "cooperative_kernel", False)
            and (
                getattr(args, "cu_fused_vdit", False)
                or getattr(args, "cu_fused_adit", False)
                or getattr(args, "cu_fused_text_encoder", False)
            )
            else []
        )
        + (["--action-pre-fused"] if getattr(args, "action_pre_fused", False) else [])
        + (["--video-pre-fused"] if getattr(args, "video_pre_fused", False) else [])
        + (["--require-fused"] if getattr(args, "require_fused", False) else [])
        + (["--text-bf16"] if not getattr(args, "text_fp8", True) else [])
        + (["--no-prompt-cache"] if getattr(args, "no_prompt_cache", False) else [])
        + (["--no-action-context-cache"] if getattr(args, "no_action_context_cache", False) else [])
        + (["--camera-alias", args.camera_alias] if getattr(args, "camera_alias", None) else [])
        + (["--rtc-debug-maxlen", str(args.rtc_debug_maxlen)] if getattr(args, "rtc_debug_maxlen", 100) != 100 else [])
    )


def _import_gateway(args: argparse.Namespace) -> None:
    from .gateway import main as gateway_main

    gateway_main(
        [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--worker-socket",
            args.worker_socket,
            "--worker-host",
            args.worker_host,
            "--worker-port",
            str(args.worker_port),
        ]
        + (["--ipc-uint8"] if args.ipc_uint8 else [])
        + (["--shm-ipc"] if args.shm_ipc else [])
        + (["--gpu-ipc"] if args.gpu_ipc else [])
        + (["--gpu-slots", str(args.gpu_slots)] if args.gpu_ipc else [])
        + (["--timing"] if getattr(args, "timing", False) else [])
        + (["--camera-alias", args.camera_alias] if getattr(args, "camera_alias", None) else [])
    )


if __name__ == "__main__":
    main()
