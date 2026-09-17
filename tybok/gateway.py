"""WebSocket gateway + image processing (Python, model-agnostic).

Implements the gateway role of plan 2 in Python: it owns the WebSocket
endpoint and the image processing (decode -> resize -> left/top padding), and
forwards model-ready tensors to the inference worker over a local socket.

The gateway learns everything it needs about the model (camera keys, resize
target, action dim) from the worker's ``describe`` message, so it works for any
registered policy backend (SmolVLA today, pi0.5 / lingbot-vla later) without
code changes.

Client protocol (JSON text frames)::

    {
      "images": {"camera1": "<base64 jpeg/png>", "camera2": "...", "camera3": "..."},
      "state":  [0.1, 0.2, ...],          # 1-D float array
      "task":   "pick up the cup",
      "mode":   "select_action"           # optional: "predict_action_chunk"
    }

Response::

    {"ok": true, "action": [..], "shape": [6]}            # select_action
    {"ok": true, "action": [[...], ...], "shape": [50, 6]}  # predict_action_chunk
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mmap
import os
import socket
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from aiohttp import web

from .logging_utils import setup_logging
from .protocol import (
    MAX_FRAME_BYTES,
    PREFIX_BYTES,
    WORKER_STARTUP_TIMEOUT,
    decode_response,
    encode_control,
    encode_request,
    encode_request_gpu,
)

log = logging.getLogger("tybok.gateway")

# aiohttp, numpy and the image pipeline are imported per function rather than at module
# level: this module also holds the ``gateway`` parser, which the CLI surface guard reads
# on a machine without the inference stack installed. The imports are one ``sys.modules``
# lookup per call, next to JPEG decodes and worker roundtrips.

# The observation prefix every camera slot lives under on the worker side; backends report
# their slots either bare (``camera1``) or already prefixed (fastwam's
# ``observation.images.image``), so both forms are normalised through this.
OBS_IMAGES = "observation.images."


class WorkerClient:
    """Async IPC client to the inference worker (one persistent connection).

    The connection is opened on first use and reused across requests: the
    worker serves each connection from a dedicated thread, so keeping the
    socket alive avoids a connect/accept/thread-spawn per request (~0.1-0.5ms)
    and lets the worker's read loop stay warm. Requests are serialised with a
    lock (the worker's inference is serialised anyway, so no concurrency is
    lost); a failed roundtrip closes and reconnects the socket.
    """

    def __init__(self, socket_path: str | None, host: str, port: int):
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self._lock = asyncio.Lock()
        self._sock: socket.socket | None = None
        self._shm_counter = 0  # ring index into the shared-memory region
        # GPU-direct keep-alive ring: slot i holds the CUDA tensors of request
        # i (the worker must import them before the slot is rewritten, which is
        # guaranteed by the bounded worker queue -- same argument as the shm
        # slots). Dropping the reference lets the caching allocator recycle.
        self._gpu_counter = 0
        self._gpu_pending: list[list[object]] = []
        # Outage tracking, so that a worker which is still starting (or restarting) costs two
        # log lines instead of one per attempt: the first failure of an outage reports the
        # reason, the remaining attempts stay silent, and the recovery reports once. Runtime
        # failures are not hidden by this -- ``_handle_one`` still logs the request traceback.
        self._failed_attempts = 0
        self._outage_started = 0.0
        self._connected_once = False

    async def connect(self) -> socket.socket:
        loop = asyncio.get_running_loop()
        if self.socket_path is not None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.setblocking(False)
            await loop.sock_connect(sock, self.socket_path)
            return sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        await loop.sock_connect(sock, (self.host, self.port))
        return sock

    async def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    async def _roundtrip(self, frame: bytes | bytearray, request_id: str) -> dict:
        async with self._lock:
            for _ in range(3):
                try:
                    if self._sock is None:
                        self._sock = await self.connect()
                    sock = self._sock
                    await self._send_all(sock, frame)
                    resp_frame = await self._read_frame(sock)
                    resp = decode_response(resp_frame)
                    if resp.get("id") != request_id:
                        raise RuntimeError(f"request id mismatch: {resp.get('id')} != {request_id}")
                except (OSError, asyncio.IncompleteReadError) as e:
                    await self._close()
                    self._failed_attempts += 1
                    if self._failed_attempts == 1:  # first failure of this outage
                        self._outage_started = time.monotonic()
                        if self._connected_once:
                            log.warning("worker request failed (%s); reconnecting ...", e)
                        else:
                            log.info("waiting for the worker to start (%s) ...", e)
                    continue
                # Success: report the recovery once. The stated wait covers the whole outage,
                # not this call's 3 retries (``_fetch_spec`` may call us many times).
                if self._failed_attempts:
                    waited = time.monotonic() - self._outage_started
                    if self._connected_once:
                        log.info(
                            "worker reachable again after %.1fs (%d failed attempts)", waited, self._failed_attempts
                        )
                    else:
                        log.info(
                            "worker is up after %.1fs (%d failed attempts while starting)",
                            waited,
                            self._failed_attempts,
                        )
                    self._failed_attempts = 0
                self._connected_once = True
                return resp
        raise RuntimeError("worker unreachable after 3 attempts")

    async def request(
        self,
        task: str,
        tensors: dict[str, np.ndarray],
        mode: str,
        noise: str | None = None,
        quantize_images: bool = False,
        shm: tuple[mmap.mmap, int, int] | None = None,
        gpu: tuple[int, int] | None = None,
        rtc: dict | None = None,
    ) -> dict:
        request_id = uuid.uuid4().hex
        if gpu is not None:
            device, slots = gpu
            slot = self._gpu_counter % slots
            self._gpu_counter += 1
            frame, keep = encode_request_gpu(
                request_id,
                task,
                tensors,
                mode=mode,
                noise=noise,
                quantize_images=quantize_images,
                device=device,
                rtc=rtc,
            )
            while len(self._gpu_pending) <= slot:
                self._gpu_pending.append([])
            self._gpu_pending[slot] = keep  # drop the previous slot's tensors
        elif shm is not None:
            region, slot_bytes, slots = shm
            slot = self._shm_counter % slots
            self._shm_counter += 1
            frame = encode_request(
                request_id,
                task,
                tensors,
                mode=mode,
                noise=noise,
                quantize_images=quantize_images,
                shm=(region, slot_bytes, slot),
                rtc=rtc,
            )
        else:
            frame = encode_request(
                request_id, task, tensors, mode=mode, noise=noise, quantize_images=quantize_images, rtc=rtc
            )
        return await self._roundtrip(frame, request_id)

    async def describe(self) -> dict:
        request_id = uuid.uuid4().hex
        frame = encode_control(request_id, "describe")
        resp = await self._roundtrip(frame, request_id)
        if not resp.get("ok"):
            raise RuntimeError(f"worker describe failed: {resp.get('error')}")
        return resp["data"]

    async def _send_all(self, sock: socket.socket, data: bytes | bytearray) -> None:
        loop = asyncio.get_running_loop()
        # Python 3.11 sock_sendall returns None on full send; 3.12+ returns
        # the number of bytes sent. Handle both.
        while data:
            n = await loop.sock_sendall(sock, data)
            if n is None:
                return
            data = data[n:]

    async def _read_frame(self, sock: socket.socket) -> bytes:
        loop = asyncio.get_running_loop()
        prefix = b""
        while len(prefix) < PREFIX_BYTES:
            chunk = await loop.sock_recv(sock, PREFIX_BYTES - len(prefix))
            if not chunk:
                raise asyncio.IncompleteReadError(prefix, PREFIX_BYTES)
            prefix += chunk
        total_len = int.from_bytes(prefix[:4], "little")
        if total_len > MAX_FRAME_BYTES:
            raise ValueError(f"frame too large: {total_len}")
        body = b""
        while len(body) < total_len - PREFIX_BYTES:
            chunk = await loop.sock_recv(sock, total_len - PREFIX_BYTES - len(body))
            if not chunk:
                raise asyncio.IncompleteReadError(body, total_len - PREFIX_BYTES)
            body += chunk
        return b"".join((prefix, body))


class Gateway:
    """WebSocket gateway. ``spec`` comes from the worker's ``describe``."""

    def __init__(
        self,
        worker: WorkerClient,
        spec: dict,
        *,
        max_inflight: int = 4,
        ipc_uint8: bool = False,
        shm: tuple[mmap.mmap, int, int] | None = None,
        gpu: tuple[int, int] | None = None,
        timing: bool = False,
        camera_alias: dict[str, str] | None = None,
    ):
        self.worker = worker
        self.spec = spec
        # Camera slots as the backend's ``describe()`` reports them: the bare slot name for
        # smolvla/pi05 (``camera1`` / ``image``), the full observation key for fastwam
        # (``observation.images.image`` -- its checkpoint declares the frame that way).
        # Clients may send either form; ``_slot_name`` normalises both to the bare name and
        # the worker always receives ``observation.images.<bare>``.
        self.camera_keys = list(spec["cameras"])
        self.image_size = tuple(spec["resize"])  # (width, height) checkpoint convention
        self.pad_mode = spec.get("pad_mode", "top-left")  # pi05 uses centered padding
        # Client camera names -> model slot names (e.g. ``{"wrist_image": "image2"}``):
        # when the slot key is absent from a client payload, the alias source is
        # decoded into the slot instead. Default empty -- clients must use the
        # exact camera names from the worker describe().
        self.camera_alias = camera_alias or {}
        # Cross-request pipelining: how many client messages may be decoded /
        # roundtripped concurrently per connection (see ``handle_ws``).
        self._max_inflight = max_inflight
        # ``--ipc-uint8``: quantise the resized [0,1] float32 images to uint8
        # for the gateway<->worker IPC (6.3MB -> 1.6MB for 2x512x512x3). The
        # 1/255 rounding changes the pixels the model sees (opt-in; the
        # bit-exact validate/e2e paths keep float32).
        self.ipc_uint8 = bool(ipc_uint8)
        # ``--shm-ipc``: (region, slot_bytes, slots) shared-memory payload
        # region, opened from the worker's describe() output.
        self._shm = shm
        # ``--gpu-ipc``: (device, slots) -- the gateway HtoD's each frame and
        # shares the CUDA buffers with the worker via cudaIpcMemHandle.
        self._gpu = gpu
        self.timing = bool(timing)  # --timing: per-request phases to stderr (like the C++ gateway)
        self._req_counter = 0

    @staticmethod
    def _slot_name(cam: str) -> str:
        """Bare slot name of a ``describe()`` camera entry (strips the observation prefix)."""
        return cam[len(OBS_IMAGES) :] if cam.startswith(OBS_IMAGES) else cam

    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode_one(raw, image_size: tuple[int, int], pad_mode: str = "top-left") -> np.ndarray:
        """Decode + resize one camera payload to a ``(C, H, W)`` float32 numpy
        tensor in ``[0, 1]``. Runs in a worker thread (releases the GIL during
        the JPEG decode and the torch resize), so it must not touch the event
        loop."""
        import numpy as np
        import torch

        from .image_utils import base64_to_tensor, decode_image_to_tensor, prepare_image_tensor

        if isinstance(raw, str):
            img = base64_to_tensor(raw)  # (3, H, W) float32 [0,1]
        elif isinstance(raw, (list, tuple)):
            arr = np.asarray(raw)
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            img = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        elif isinstance(raw, (bytes, bytearray)):
            img = decode_image_to_tensor(bytes(raw))
        else:
            raise ValueError(f"unsupported image payload: {type(raw).__name__}")
        img = prepare_image_tensor(img, image_size, pad_mode=pad_mode)  # (C, H, W)
        return img.numpy()

    async def _process_images(self, raw_images: dict) -> tuple[dict[str, np.ndarray], list[str]]:
        """Decode + resize each camera image off the event loop, in parallel.

        The JPEG decode is the gateway's biggest per-request cost (~5-8ms per
        camera on CPU); running it synchronously in the aiohttp loop would stall
        every other client. Each camera is decoded in a thread-pool worker
        (``asyncio.to_thread``; PIL's JPEG decode and the torch resize release
        the GIL, so cameras genuinely run in parallel on multicore hosts).
        """
        tensors: dict[str, np.ndarray] = {}
        missing: list[str] = []

        def payload_for(cam: str) -> str | None:
            """Client key for a slot: the reported name, its bare observation key, or an alias."""
            bare = self._slot_name(cam)
            for candidate in (cam, bare, f"{OBS_IMAGES}{bare}"):
                if candidate in raw_images:
                    return candidate
            for src, dst in self.camera_alias.items():
                if dst == cam and src in raw_images:
                    return src
            return None

        async def _decode(cam: str):
            src = payload_for(cam)
            if src is None:
                missing.append(cam)
                return
            raw = raw_images[src]
            arr = await asyncio.to_thread(self._decode_one, raw, self.image_size, self.pad_mode)
            tensors[f"{OBS_IMAGES}{self._slot_name(cam)}"] = arr  # always under the model slot key

        await asyncio.gather(*(_decode(cam) for cam in self.camera_keys))
        return tensors, missing

    # ------------------------------------------------------------------ #
    async def _handle_one(
        self,
        ws: web.WebSocketResponse,
        payload: dict,
    ) -> None:
        """Decode + infer + respond for a single client message.

        Runs as a per-message task so the *next* message's decode can overlap
        this one's worker roundtrip (cross-request pipelining): the aiohttp
        read loop keeps consuming messages while this task awaits the worker.
        The worker-client lock (FIFO) serialises the actual requests, so
        responses leave in message order.
        """
        import numpy as np

        t0 = time.perf_counter()
        wait_decode = 0.0
        try:
            task = payload.get("task", "")
            mode = payload.get("mode", "select_action")
            noise = payload.get("noise")  # "zeros" | None (deterministic testing)
            raw_images = payload.get("images", {})
            state = np.asarray(payload.get("state", []), dtype=np.float32)

            rtc = payload.get("rtc")
            if rtc is not None and not isinstance(rtc, dict):
                raise ValueError("rtc must be a JSON object, e.g. {'inference_delay': 2, ...}")
            if rtc and not self.spec.get("rtc", False):
                await ws.send_json(
                    {"ok": False, "error": "RTC is not enabled on this deployment (start the worker with --rtc)"}
                )
                return

            tensors, missing = await self._process_images(raw_images)
            wait_decode = (time.perf_counter() - t0) * 1e3
            if not tensors:
                await ws.send_json({"ok": False, "error": f"no images received (expected: {self.camera_keys})"})
                return
            tensors["observation.state"] = state

            resp = await self.worker.request(
                task,
                tensors,
                mode=mode,
                noise=noise,
                quantize_images=self.ipc_uint8,
                shm=self._shm,
                gpu=self._gpu,
                rtc=rtc or None,
            )
            if resp.get("ok"):
                message = {
                    "ok": True,
                    "model": self.spec.get("model_type"),
                    "mode": mode,
                    "action": resp["data"]["action"],
                    "shape": resp["data"]["shape"],
                    "missing_cameras": missing,
                }
                # RTC deployments also expose the model-space chunk so the client
                # can seed its next request's prev_chunk_left_over.
                if "action_normalized" in resp.get("data", {}):
                    message["action_normalized"] = resp["data"]["action_normalized"]
                await ws.send_json(message)
            else:
                await ws.send_json({"ok": False, "error": resp.get("error")})
            if self.timing:
                total = (time.perf_counter() - t0) * 1e3
                # Same format as the C++ gateway's --timing line.
                log.info(f"[timing] req=py-{self._req_counter} wait-decode={wait_decode:.2f}ms total={total:.2f}ms")
                self._req_counter += 1
        except Exception as e:  # noqa: BLE001
            log.exception("request failed")
            try:
                await ws.send_json({"ok": False, "error": str(e)})
            except Exception:  # noqa: BLE001 - connection already gone
                pass

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=64 * 1024 * 1024)
        await ws.prepare(request)
        log.info("client connected")
        # Per-message tasks, bounded: at most ``_MAX_INFLIGHT`` messages are
        # decoded/roundtripped concurrently per connection (a pipelining client
        # sends the next observation before the previous action arrives, so the
        # decode + encode of message N+1 overlaps the inference of N). The
        # worker-client lock keeps the requests -- and hence the responses -- in
        # order.
        inflight: list[asyncio.Task] = []
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
                if msg.type != WSMsgType.TEXT:
                    await ws.send_json({"ok": False, "error": "only JSON text frames are supported"})
                    continue
                try:
                    payload = json.loads(msg.data)
                except Exception as e:  # noqa: BLE001
                    await ws.send_json({"ok": False, "error": str(e)})
                    continue
                inflight.append(asyncio.create_task(self._handle_one(ws, payload)))
                # bound the in-flight tasks: wait for the oldest to finish
                while len(inflight) >= self._max_inflight:
                    # ``wait`` returns ``(done, pending)`` and pending is a set, so rebuild the list
                    # we keep appending to; re-check ``done``, a task can finish while we are here.
                    _, pending = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                    inflight = [t for t in pending if not t.done()]
            # drain the remaining tasks before closing
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
        finally:
            for t in inflight:
                if not t.done():
                    t.cancel()
        log.info("client disconnected")
        return ws

    async def handle_health(self, request: web.Request) -> web.Response:
        from aiohttp import web

        return web.json_response({"status": "ok", "model": self.spec.get("model_type"), "cameras": self.camera_keys})

    # ------------------------------------------------------------------ #
    def app(self) -> web.Application:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/ws", self.handle_ws)
        app.router.add_get("/health", self.handle_health)
        return app


async def _fetch_spec(worker: WorkerClient, timeout: float = WORKER_STARTUP_TIMEOUT, delay: float = 1.0) -> dict:
    """Ask the worker for the model spec, retrying until it is up (or ``timeout`` elapses).

    The budget is generous (``protocol.WORKER_STARTUP_TIMEOUT``) because an old/slow CPU can
    spend minutes in weight loading, and the retries are silent -- the whole wait costs two
    log lines (see ``WorkerClient``). This deadline is a backstop for a worker that never
    comes up; a worker that *dies* at startup is caught by ``tybok serve``'s early-exit check.
    """
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return await worker.describe()
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(delay)
    raise RuntimeError(f"worker unreachable after {timeout:.0f}s (describe failed): {last_err}")


def _open_shm_region(path: str, total: int) -> mmap.mmap:
    """Open (and size-check) the worker's shared-memory payload region.

    NB: ``access=`` (not a positional third arg) -- ``mmap.ACCESS_WRITE`` as the
    third positional argument lands in ``flags`` and equals ``MAP_PRIVATE``,
    which silently breaks cross-process visibility.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        if os.fstat(fd).st_size < total:
            raise RuntimeError(f"shm region {path} too small ({os.fstat(fd).st_size} < {total})")
        return mmap.mmap(fd, total, access=mmap.ACCESS_WRITE)
    finally:
        os.close(fd)


def _shm_from_spec(spec: dict) -> tuple[mmap.mmap, int, int]:
    shm = spec.get("shm")
    if not shm:
        raise RuntimeError(
            "--shm-ipc requested but the worker advertises no shm region (start the worker with --shm-ipc)"
        )
    slot_bytes = int(shm["slot_bytes"])
    slots = int(shm["slots"])
    region = _open_shm_region(shm["path"], slot_bytes * slots)
    return region, slot_bytes, slots


def _parse_camera_alias(spec: str | None) -> dict[str, str] | None:
    """Parse ``--camera-alias SRC=DST[,SRC=DST...]`` into a ``{client: slot}`` dict.

    Kept here rather than imported from ``worker`` so the gateway stays free of the
    engine/torch import chain; the worker-side twin is ``worker.parse_camera_alias``.
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Policy WebSocket gateway")
    parser.add_argument("--host", default="0.0.0.0", help="gateway bind host")
    parser.add_argument("--port", type=int, default=8765, help="gateway bind port")
    parser.add_argument("--worker-socket", default=None, help="worker unix socket path")
    parser.add_argument("--worker-host", default="127.0.0.1", help="worker TCP host")
    parser.add_argument("--worker-port", type=int, default=5555, help="worker TCP port")
    parser.add_argument(
        "--ipc-uint8",
        action="store_true",
        help="quantise the resized [0,1] images to uint8 for the gateway<->worker IPC "
        "(4x smaller payload; 1/255 pixel rounding -- opt-in, breaks bit-exactness)",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=4,
        help="per-connection pipelined messages decoded ahead of the worker response (default 4)",
    )
    parser.add_argument(
        "--shm-ipc",
        action="store_true",
        help="carry the ~6MB tensor payload over a shared-memory region instead of the "
        "socket (the socket then carries only the small JSON header; opt-in, byte-identical)",
    )
    parser.add_argument(
        "--gpu-ipc",
        action="store_true",
        help="GPU-direct IPC: HtoD the frame on the gateway and share the CUDA buffers "
        "with the worker via cudaIpcMemHandle (socket carries only the small JSON "
        "header; opt-in, byte-identical; requires CUDA, mutually exclusive with --shm-ipc)",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=8,
        help="gateway CUDA keep-alive ring depth for --gpu-ipc (default 8; the worker is "
        "bounded to ~3 in-flight frames, so 8 is safe)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="log one [timing] line per request (decode wait + total round trip), like the "
        "C++ gateway. Runtime and per-request, so it includes the gateway -> worker hop; "
        "the worker-side --profile is the complementary startup one-shot that breaks the "
        "inference itself down by phase",
    )
    parser.add_argument(
        "--camera-alias",
        default=None,
        metavar="SRC=DST[,SRC=DST...]",
        help="gateway: decode client camera SRC into the checkpoint image slot DST (an image "
        "key of the model config.json, e.g. image2; the slots are the camera names the "
        "worker reports in describe()). Applied only when DST is absent from the payload, "
        "so clients may still send the exact slot names. Several pairs go into one "
        "comma-separated value: "
        "--camera-alias left_wrist_image=image2,right_wrist_image=image3",
    )
    args = parser.parse_args(argv)

    setup_logging()

    worker = WorkerClient(args.worker_socket, args.worker_host, args.worker_port)
    # fetch the model spec on a short-lived loop first; web.run_app below
    # creates its own event loop, so it must not be nested inside asyncio.run
    spec = asyncio.run(_fetch_spec(worker))
    log.info("worker model spec: %s", spec)
    shm = _shm_from_spec(spec) if args.shm_ipc else None
    if shm is not None:
        log.info("shared-memory IPC region open (%dx%dMiB)", shm[2], shm[1] >> 20)
    gpu = None
    if args.gpu_ipc:
        if args.shm_ipc:
            raise RuntimeError("--shm-ipc and --gpu-ipc are mutually exclusive")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-ipc requires a CUDA device")
        torch.cuda.init()  # fail fast + absorb the one-time context cost at startup
        gpu = (torch.cuda.current_device(), args.gpu_slots)
        log.info("GPU-direct IPC enabled (device %d, %d keep-alive slots)", gpu[0], gpu[1])
    camera_alias = _parse_camera_alias(args.camera_alias)
    if camera_alias:
        log.info("camera alias: %s", camera_alias)
    gateway = Gateway(
        worker,
        spec,
        max_inflight=args.max_inflight,
        ipc_uint8=args.ipc_uint8,
        shm=shm,
        gpu=gpu,
        timing=args.timing,
        camera_alias=camera_alias,
    )

    def _announce_deployment(banner: str) -> None:
        """aiohttp calls this once the TCP site is actually listening (``print=``)."""
        log.info(banner.rstrip())
        log.info(
            f"deployment complete: model={spec.get('model_type')} cameras={gateway.camera_keys} "
            f"host={args.host} port={args.port} ws=ws://{args.host}:{args.port}/ws "
            f"health=http://{args.host}:{args.port}/health"
        )

    from aiohttp import web

    web.run_app(gateway.app(), host=args.host, port=args.port, print=_announce_deployment)


if __name__ == "__main__":
    main()
