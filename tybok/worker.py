"""Inference worker process (model-agnostic).

Owns the model and serves inference requests from the gateway over a local
Unix domain socket (or TCP). Request/response framing is defined in
:mod:`tybok.protocol`; the engine is created through the model registry
(:mod:`tybok.registry`), so the same worker binary serves SmolVLA today
and pi0.5 / lingbot-vla backends later.

The worker loads the engine once at startup (weights stay resident), then
serves ``infer`` requests. Inference is serialized with a lock: engines are
single-stream (batch size 1), and requests queue up behind it.
"""

from __future__ import annotations

import argparse
import json
import logging
import mmap
import os
import queue
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

from .logging_utils import setup_logging
from .protocol import MAX_FRAME_BYTES, PREFIX_BYTES, decode_request, decode_rtc, encode_response
from .registry import create_engine

log = logging.getLogger("tybok.worker")


def parse_camera_alias(spec: str | None) -> dict[str, str] | None:
    """Parse ``--camera-alias SRC=DST[,SRC=DST...]`` into a ``{client: slot}`` dict.

    SRC is the client-side camera key, DST is the checkpoint image slot. The flag takes a
    single comma-separated value (not ``append``): a repeated flag would silently keep
    only its last value in most shells.
    """
    if not spec:
        return None
    alias: dict[str, str] = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"--camera-alias expects SRC=DST, got {pair!r}")
        src, dst = (p.strip() for p in pair.split("=", 1))
        if not src or not dst:
            raise ValueError(f"--camera-alias expects SRC=DST, got {pair!r}")
        if src in alias and alias[src] != dst:
            raise ValueError(f"--camera-alias maps {src!r} twice to different targets")
        alias[src] = dst
    return alias


SHM_SLOTS = 8  # ring depth; the worker is bounded to ~3 in-flight frames, so
# a slot written for request N is never read again by the time N+8 is written


def _open_shm_region(path: str, total: int) -> mmap.mmap:
    """Create/truncate ``path`` to ``total`` bytes and map it writable+shared.

    NB: ``access=`` (not a positional third arg) -- ``mmap.ACCESS_WRITE`` as the
    third positional argument lands in ``flags`` and equals ``MAP_PRIVATE``,
    which silently breaks cross-process visibility.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, total)
        return mmap.mmap(fd, total, access=mmap.ACCESS_WRITE)
    finally:
        os.close(fd)


def _read_frame(conn: socket.socket) -> bytearray | None:
    """Read one length-prefixed frame; returns None on clean EOF.

    The frame is recv'd straight into a preallocated bytearray (one allocation
    per frame): the old ``prefix + body += chunk`` accumulation + final concat
    measured ~2.7ms on a 6.3MB frame vs ~1.2ms here. The returned bytearray is
    accepted everywhere a frame goes (``decode_request`` / ``_peek_request_id``
    use buffer objects, and the numpy views it creates keep it alive).
    """
    # read the 8-byte prefix into a small local first (frame size is unknown)
    prefix = b""
    while len(prefix) < PREFIX_BYTES:
        chunk = conn.recv(PREFIX_BYTES - len(prefix))
        if not chunk:
            return None
        prefix += chunk
    total_len = struct.unpack("<I", prefix[:4])[0]
    if total_len > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {total_len}")
    buf = bytearray(total_len)
    buf[:PREFIX_BYTES] = prefix
    view = memoryview(buf)
    off = PREFIX_BYTES
    while off < total_len:
        n = conn.recv_into(view[off:], total_len - off)
        if n == 0:
            return None
        off += n
    return buf


class InferenceWorker:
    def __init__(
        self,
        checkpoint_dir: str,
        model_type: str | None = None,
        device: str = "auto",
        compile_model: bool = False,
        graph: bool = False,
        graph_cameras: tuple[int, ...] | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        sampler: str = "euler",
        cache_expert_prefix_kv: bool = True,
        cache_prompt: bool = True,
        profile: bool = False,
        overlap: bool = True,
        tl_fused_vit: bool = False,
        tl_llm_flash_attn: bool = False,
        tl_llm_fused_attn: bool = False,
        tl_vit_oproj: bool = False,
        tl_fused_expert: bool = False,
        tl_fp8_llm_mlp: bool = False,
        tl_fp8_expert_mlp: bool = False,
        vit_mlp_dtype: str | None = None,
        skip_empty_images: bool = False,
        pad_free: bool = False,
        pack_qkv: bool = False,
        video_fp8: bool = True,
        action_fp8: bool = True,
        text_fp8: bool = True,
        action_fused: bool = False,
        action_fused_split: bool = False,
        action_pre_fused: bool = False,
        video_pre_fused: bool = False,
        video_fused: bool = False,
        video_fused_split: bool = False,
        action_context_cache: bool = True,
        text_fused: bool = False,
        text_fused_split: bool = False,
        require_fused: bool = False,
        camera_alias: dict[str, str] | None = None,
        shm_ipc: bool = False,
        tokenizer_dir: str | None = None,
        text_encoder_dir: str | None = None,
        vae_dir: str | None = None,
        text_encoder_device: str = "cpu",
        text_emb_cpu: bool = True,
        rtc_config: dict | None = None,
    ):
        kwargs: dict[str, Any] = {
            "compile_model": compile_model,
            "graph": graph,
            "graph_cameras": graph_cameras,
            "tl_fused_vit": tl_fused_vit,
            "tl_llm_flash_attn": tl_llm_flash_attn,
            "tl_llm_fused_attn": tl_llm_fused_attn,
            "tl_vit_oproj": tl_vit_oproj,
            "tl_fused_expert": tl_fused_expert,
            "tl_fp8_llm_mlp": tl_fp8_llm_mlp,
            "tl_fp8_expert_mlp": tl_fp8_expert_mlp,
            "vit_mlp_dtype": vit_mlp_dtype,
            "skip_empty_images": skip_empty_images,
            "pad_free": pad_free,
            "pack_qkv": pack_qkv,
            "action_context_cache": action_context_cache,
            "cache_prompt": cache_prompt,
            "text_fused": text_fused,
            "num_steps": num_steps,
            "seed": seed,
            "sampler": sampler,
            "cache_expert_prefix_kv": cache_expert_prefix_kv,
            "profile": profile,
            "overlap": overlap,
        }
        if tokenizer_dir is not None:
            kwargs["tokenizer_dir"] = tokenizer_dir
        if text_encoder_dir is not None:
            kwargs["text_encoder_dir"] = text_encoder_dir
        if vae_dir is not None:
            kwargs["vae_dir"] = vae_dir
        if text_encoder_device:
            kwargs["text_encoder_device"] = text_encoder_device
        if not text_emb_cpu:
            kwargs["text_emb_cpu"] = False
        if not video_fp8:
            kwargs["video_fp8"] = False
        if not action_fp8:
            kwargs["action_fp8"] = False
        if action_fused:
            kwargs["action_fused"] = True
        if action_fused_split:
            kwargs["action_fused_split"] = True
        if action_pre_fused:
            kwargs["action_pre_fused"] = True
        if video_pre_fused:
            kwargs["video_pre_fused"] = True
        if video_fused:
            kwargs["video_fused"] = True
        if video_fused_split:
            kwargs["video_fused_split"] = True
        if text_fused_split:
            kwargs["text_fused_split"] = True
        if require_fused:
            kwargs["require_fused"] = True
        if not text_fp8:
            kwargs["text_fp8"] = False
        if rtc_config is not None:
            kwargs["rtc_config"] = rtc_config
        if camera_alias:
            kwargs["camera_alias"] = camera_alias
        self.engine = create_engine(
            checkpoint_dir,
            model_type=model_type,
            device=device,
            **kwargs,
        )
        if rtc_config is not None and not self.engine.supports_rtc():
            raise ValueError(
                "--rtc was requested but the model backend does not support Real-Time Chunking (only pi05 does)"
            )
        self._lock = threading.Lock()
        self._shutdown = False
        self.shm_ipc = bool(shm_ipc)
        # (mmap, slot_bytes, slots, path) -- created in serve_socket once the
        # socket path is known; advertised via describe() so the gateway can
        # open the same region.
        self._shm: tuple[mmap.mmap, int, int, str] | None = None

    def describe(self) -> dict:
        spec = self.engine.describe()
        if self._shm is not None:
            region, slot_bytes, slots, path = self._shm
            spec["shm"] = {"path": path, "slot_bytes": slot_bytes, "slots": slots}
        return spec

    # ------------------------------------------------------------------ #
    def _setup_shm(self, socket_path: str | None, port: int | None) -> None:
        """Create the shared-memory payload region (see protocol.encode_request)."""
        slot_bytes = self._shm_slot_bytes()
        path = f"{socket_path}.shm" if socket_path else f"/dev/shm/vla_shm_{port}.shm"
        region = _open_shm_region(path, slot_bytes * SHM_SLOTS)
        self._shm = (region, slot_bytes, SHM_SLOTS, path)
        log.info(f"shared-memory IPC region {path} ({SHM_SLOTS}x{slot_bytes >> 20}MiB)")

    def _shm_slot_bytes(self) -> int:
        """Largest inline float32 payload for this model (cameras x resize + state + margin)."""
        spec = self.engine.describe()
        n_cams = max(1, len(spec.get("cameras", [])))
        w, h = spec.get("resize", [512, 512])
        payload = n_cams * 3 * h * w * 4 + 8 * 4 + 4096
        return (payload + (1 << 20) - 1) & ~((1 << 20) - 1)

    # ------------------------------------------------------------------ #
    def handle_frame(self, frame: bytes | bytearray) -> bytes:
        # numpy / torch are imported here, not at module level: this module also builds
        # ``main``'s parser, which the CLI surface guard reads on a machine without the
        # inference stack (one ``sys.modules`` lookup per request).
        import numpy as np
        import torch

        shm = (self._shm[0], self._shm[1]) if self._shm is not None else None
        request_id, mode, task, tensors, noise, quantized = decode_request(frame, shm=shm)
        if task and not task.endswith("\n"):
            task = f"{task}\n"

        rtc = decode_rtc(frame)
        rtc_kwargs = {}
        if rtc:
            prev = rtc.get("prev_chunk_left_over")
            if prev is not None:
                rtc_kwargs["prev_chunk_left_over"] = np.asarray(prev, dtype=np.float32)
            if "inference_delay" in rtc:
                rtc_kwargs["inference_delay"] = int(rtc["inference_delay"])
            if "execution_horizon" in rtc:
                rtc_kwargs["execution_horizon"] = int(rtc["execution_horizon"])

        rtc_on = bool(self.describe().get("rtc", False))
        if rtc_kwargs and not rtc_on:
            raise ValueError(
                "rtc fields were sent but the worker was started without --rtc "
                "(the model backend does not apply RTC guidance)"
            )

        # tensors are named observation keys; split images vs state (numpy views
        # for the socket/shm paths, CUDA tensors for the GPU-direct path)
        images: dict[str, np.ndarray | torch.Tensor] = {}
        state: np.ndarray | None = None
        for name, arr in tensors.items():
            if name.startswith("observation.images."):
                if quantized:
                    if isinstance(arr, torch.Tensor):
                        # GPU-direct path: the gateway quantized on CPU then
                        # HtoD'd; scale back on device.
                        arr = arr.to(torch.float32) / 255.0
                    else:
                        # the gateway rounded [0,1] floats to uint8; scale back
                        arr = arr.astype(np.float32) / 255.0
                images[name[len("observation.images.") :]] = arr
            elif name == "observation.state":
                state = arr

        if state is None:
            raise ValueError("request missing observation.state")
        if not images:
            raise ValueError("request missing observation images")

        noise_tensor = None
        if noise == "zeros":
            spec = self.engine.describe()
            noise_tensor = torch.zeros(
                1, spec["chunk_size"], spec["action_dim"], dtype=torch.float32, device=self.engine.device
            )

        frame_dict = self.engine.make_frame(images, state, task)
        with self._lock:
            # One timing line per completed inference (engine call only; the JSON encode of
            # the action chunk below is deliberately outside the measurement).
            norm = None
            t0 = time.perf_counter()
            if mode == "predict_action_chunk":
                if rtc_on:
                    # also return the model-space (normalized) chunk so a client
                    # can seed its RTC queue with the unconsumed tail.
                    out, norm = self.engine.predict_action_chunk(
                        frame_dict, noise=noise_tensor, return_normalized=True, **rtc_kwargs
                    )
                else:
                    out = self.engine.predict_action_chunk(frame_dict, noise=noise_tensor)
            else:
                out = self.engine.select_action(frame_dict, noise=noise_tensor)
            infer_ms = (time.perf_counter() - t0) * 1e3
            data = {"mode": mode, "action": out.tolist(), "shape": list(out.shape)}
            if norm is not None:
                data["action_normalized"] = norm.tolist()

        log.info(
            f"[inference] req={request_id} model={self.engine.describe().get('model_type')} "
            f"mode={mode} cameras={len(images)} {infer_ms:.2f}ms"
        )
        return encode_response(request_id, True, data)

    def handle_ping(self, request_id: str) -> bytes:
        return encode_response(request_id, True, {"pong": True})

    def handle_describe(self, request_id: str) -> bytes:
        return encode_response(request_id, True, self.describe())

    # ------------------------------------------------------------------ #
    def serve_socket(self, socket_path: str | None, host: str | None, port: int | None) -> None:
        if self.shm_ipc:
            self._setup_shm(socket_path, port)
        if socket_path is not None:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(socket_path)
            server.listen(16)
            endpoint = socket_path
            log.info(f"listening on unix socket {socket_path}")
        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host or "127.0.0.1", port or 5555))
            server.listen(16)
            endpoint = f"{host or '127.0.0.1'}:{port or 5555}"
            log.info(f"listening on {endpoint}")
        spec = self.engine.describe()
        log.info(
            f"worker deployment complete: model={spec.get('model_type')} "
            f"device={self.engine.device} cameras={spec.get('cameras')} "
            f"endpoint={endpoint} graph={bool(getattr(self.engine, 'graph_enabled', False))}"
        )

        while not self._shutdown:
            try:
                conn, _ = server.accept()
            except OSError:
                break
            t = threading.Thread(target=self._handle_conn, args=(conn,), daemon=True)
            t.start()
        server.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        """Serve one connection: a reader thread recvs frames into a bounded
        queue while this thread processes them, so a pipelined gateway (which
        sends the next request's frame before reading the current response) has
        its recv + parse overlap the current inference instead of serialising
        behind it. ``maxsize`` bounds the backlog (backpressure on the sender).
        """
        frames: "queue.Queue[bytes | bytearray | None]" = queue.Queue(maxsize=2)

        def reader() -> None:
            try:
                while True:
                    frame = _read_frame(conn)
                    frames.put(frame)  # None on clean EOF
                    if frame is None:
                        break
            except (OSError, ValueError):
                frames.put(None)

        threading.Thread(target=reader, daemon=True).start()
        with conn:
            while True:
                frame = frames.get()
                if frame is None:
                    break
                try:
                    resp = self._process_frame(frame)
                except Exception as e:  # noqa: BLE001 - report any failure back to the gateway
                    request_id = self._peek_request_id(frame)
                    resp = encode_response(request_id, False, {}, error=str(e))
                try:
                    conn.sendall(resp)
                except OSError:
                    break
                if self._shutdown:
                    break

    @staticmethod
    def _peek_request_id(frame: bytes | bytearray) -> str:
        try:
            _, header_len = struct.unpack("<II", frame[:PREFIX_BYTES])
            header = json.loads(frame[PREFIX_BYTES : PREFIX_BYTES + header_len].decode("utf-8"))
            return header.get("id", "")
        except Exception:  # noqa: BLE001
            return ""

    def _process_frame(self, frame: bytes | bytearray) -> bytes:
        """Handle one request frame (decode + infer + encode the response)."""
        _, header_len = struct.unpack("<II", frame[:PREFIX_BYTES])
        header = json.loads(frame[PREFIX_BYTES : PREFIX_BYTES + header_len].decode("utf-8"))
        mtype = header.get("type", "infer")
        request_id = header.get("id", "")
        if mtype == "ping":
            return self.handle_ping(request_id)
        if mtype == "describe":
            return self.handle_describe(request_id)
        if mtype == "shutdown":
            resp = encode_response(request_id, True, {"bye": True})
            self._shutdown = True
            return resp
        return self.handle_frame(frame)


def _parse_graph_cameras(s: str | None) -> tuple[int, ...] | None:
    if not s:
        return None
    counts = tuple(int(x) for x in s.split(",") if x.strip())
    if not counts or any(n <= 0 for n in counts):
        raise argparse.ArgumentTypeError(f"invalid --graph-cameras {s!r}")
    return counts


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Policy inference worker")
    parser.add_argument("--model", required=True, help="path to the model checkpoint dir")
    parser.add_argument(
        "--model-type", default=None, help="backend override (default: the checkpoint config.json 'type')"
    )
    parser.add_argument("--socket", default=None, help="unix socket path (default: use TCP)")
    parser.add_argument("--host", default="127.0.0.1", help="TCP host")
    parser.add_argument("--port", type=int, default=5555, help="TCP port")
    parser.add_argument("--device", default="auto", help="torch device (auto/cuda/cpu)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the sampling loop (changes numerics)")
    parser.add_argument("--graph", action="store_true", help="capture the sampling loop as a CUDA graph (bit-exact)")
    parser.add_argument(
        "--graph-cameras",
        default=None,
        help="comma-separated camera counts whose CUDA-graph shape to pre-capture at startup, "
        "e.g. '2,3' (default: the configured camera count; other counts are captured "
        "lazily). This selects which *prefix-length buckets* are ready -- each camera is its "
        "own single-image ViT forward, so the count changes the sequence length, not a batch "
        "dimension; it is not a ViT batch setting",
    )
    parser.add_argument(
        "--tl-fused-vit",
        action="store_true",
        help="pi0.5: fuse the ViT attention, MLP and projector into Triton kernels",
    )
    parser.add_argument(
        "--tl-llm-flash-attn",
        action="store_true",
        help="pi0.5: VLM prefill attention as a Triton GQA flash kernel (q/k/v concatenated into one GEMM)",
    )
    parser.add_argument(
        "--tl-llm-fused-attn",
        action="store_true",
        help="smolvla: fully fuse the LLM prefill attention in Triton "
        "(RMSNorm + q/k/v + RoPE + KV-cache write + GQA flash)",
    )
    parser.add_argument(
        "--tl-vit-oproj",
        action="store_true",
        help="smolvla: vision out_proj as one Triton kernel; q/k/v are concatenated "
        "into one GEMM (vision MLP is not fused)",
    )
    parser.add_argument(
        "--tl-fused-expert",
        action="store_true",
        help="pi0.5 / smolvla: fuse the denoising-expert attention and MLP into Triton kernels",
    )
    parser.add_argument(
        "--vit-mlp-dtype",
        default=None,
        choices=["fp16", "bf16"],
        help="pi0.5: dtype of the ViT MLP GEMMs (fc1/fc2); other ViT ops stay fp32; ignored under --tl-fused-vit",
    )
    parser.add_argument(
        "--skip-empty-cams",
        action="store_true",
        help="pi0.5: skip the ViT forward for the empty camera slots and pad the prefix "
        "with zeros instead (bit-exact; the check sits in embed_prefix, before the ViT, "
        "so it also applies under --tl-fused-vit). Not available on the --graph path, "
        "which bakes all-True masks at capture -- use --pad-free there",
    )
    parser.add_argument(
        "--pad-free",
        action="store_true",
        help="pi0.5: padding-free VLM prefill (drop empty-camera and language-pad tokens from the prefix)",
    )
    parser.add_argument(
        "--tl-fp8-llm-mlp",
        action="store_true",
        help="pi0.5: run the VLM prefill MLP as a fused Triton W8A8 fp8 GEMM (requires an fp8-capable GPU)",
    )
    parser.add_argument(
        "--tl-fp8-expert-mlp",
        action="store_true",
        help="pi0.5: run the denoising-expert MLP as a fused Triton W8A8 fp8 chain (requires an fp8-capable GPU)",
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
        "--no-prompt-cache",
        action="store_true",
        help="fastwam: re-encode the task prompt (UMT5) on every request instead of reusing "
        "the single-entry prompt memo. Bit-identical either way; use it for profiling / "
        "benchmarking, where the cached path would hide the text-encoder cost.",
    )
    parser.add_argument(
        "--no-action-context-cache",
        action="store_true",
        help="fastwam: recompute the action expert's cross-attention context (text embedding "
        "+ per-layer k/v) on every denoising step instead of once per chunk. Bit-exact "
        "either way; kept for A/B timing.",
    )
    parser.add_argument(
        "--cu-fused-text-encoder",
        action="store_true",
        help="fastwam: run the two UMT5 sublayers (self-attention + FFN) with fused CUDA "
        "kernels (needs --text-encoder-device cuda + fp8 + a CUDA device); with "
        "--cooperative-kernel uses cooperative launches",
    )
    parser.add_argument(
        "--camera-alias",
        default=None,
        metavar="SRC=DST[,SRC=DST...]",
        help="rename client camera SRC to checkpoint image slot DST: the client sends SRC "
        "(e.g. left_wrist_image), the engine serves it in the slot DST that the model "
        "config.json expects (an image key, e.g. image2). Applied only when DST is "
        "absent from the frame. Several pairs go into one comma-separated value: "
        "--camera-alias left_wrist_image=image2,right_wrist_image=image3",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Euler denoising step count (default: checkpoint config, usually 10); "
        "fewer steps change the output vs the reference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the denoising-noise RNG for reproducible action sequences (default: global torch RNG)",
    )
    parser.add_argument(
        "--sampler",
        default="euler",
        choices=["euler", "heun"],
        help="denoising sampler: euler (reference, bit-exact) or heun "
        "(2nd-order predictor-corrector; N steps = 2N velocity evals, "
        "not bit-exact against the reference)",
    )
    parser.add_argument(
        "--no-expert-prefix-kv-cache",
        action="store_true",
        help="smolvla: disable the expert prefix-KV projection cache (recompute the fp32 "
        "expert cross-attention projections of the prefix KV every denoising step; "
        "bit-exact either way, just slower)",
    )
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overlap the prefix prefill with the first denoising step (smolvla/pi05: "
        "`prefill_layer` x `step0_layer`; fastwam: `prefill_video_layer` x the step-0 "
        "`action_layer`), so the step-0 layers hide inside the prefill window; `--no-overlap` "
        "runs the two strictly in sequence instead. Bit-exact either way (same kernels, "
        "different launch order). On by default wherever it applies: smolvla/pi05 only on the "
        "--graph path (CUDA + euler), fastwam on eager as well; --compile disables it. The "
        "eager-mode gain is platform-dependent and has not been broadly tested",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run a one-shot latency profile at startup and log it through the backend "
        "logger: wall time (predict_action_chunk ms/iter + Hz) and, where the backend "
        "implements it, a per-phase breakdown. Startup and worker-side, so it excludes the "
        "gateway/IPC hop; the gateway's --timing is the complementary per-request runtime "
        "counter",
    )
    parser.add_argument(
        "--shm-ipc",
        action="store_true",
        help="share the ~6MB tensor payload with the gateway over a shared-memory "
        "region (the socket then carries only the small JSON header; opt-in, "
        "byte-identical to the socket path)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="override the PaliGemma tokenizer dir (default: resolved from the checkpoint)",
    )
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
        default="cpu",
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
    args = parser.parse_args(argv)

    rtc_config = None
    if args.rtc:
        rtc_config = {
            "enabled": True,
            "prefix_attention_schedule": args.rtc_schedule.upper(),
            "max_guidance_weight": args.rtc_max_guidance_weight,
            "execution_horizon": args.rtc_execution_horizon,
            "debug": args.rtc_debug,
            "debug_maxlen": args.rtc_debug_maxlen,
        }

    worker = InferenceWorker(
        args.model,
        model_type=args.model_type,
        device=args.device,
        compile_model=args.compile,
        graph=args.graph,
        graph_cameras=_parse_graph_cameras(args.graph_cameras),
        num_steps=args.steps,
        seed=args.seed,
        sampler=args.sampler,
        cache_expert_prefix_kv=not args.no_expert_prefix_kv_cache,
        cache_prompt=not args.no_prompt_cache,
        profile=args.profile,
        overlap=args.overlap,
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
        video_fp8=args.video_fp8,
        action_fp8=args.action_fp8,
        text_fp8=args.text_fp8,
        action_fused=args.cu_fused_adit,
        action_fused_split=(args.cu_fused_adit and not args.cooperative_kernel),
        action_pre_fused=args.action_pre_fused,
        video_pre_fused=args.video_pre_fused,
        video_fused=args.cu_fused_vdit,
        video_fused_split=(args.cu_fused_vdit and not args.cooperative_kernel),
        action_context_cache=not args.no_action_context_cache,
        text_fused=args.cu_fused_text_encoder,
        text_fused_split=(args.cu_fused_text_encoder and not args.cooperative_kernel),
        require_fused=args.require_fused,
        camera_alias=parse_camera_alias(args.camera_alias),
        shm_ipc=args.shm_ipc,
        tokenizer_dir=args.tokenizer_dir,
        text_encoder_dir=args.text_encoder_dir,
        vae_dir=args.vae_dir,
        text_encoder_device=args.text_encoder_device,
        text_emb_cpu=args.text_emb_cpu,
        rtc_config=rtc_config,
    )
    worker.serve_socket(
        args.socket, args.host if args.socket is None else None, args.port if args.socket is None else None
    )


if __name__ == "__main__":
    main()
