"""Gateway <-> worker IPC protocol.

Plan 2 splits the deployment into two processes; the gateway owns the
WebSocket endpoint and the image processing, the worker owns the model. They
talk over a local Unix domain socket (or TCP) with a simple length-prefixed
framing that avoids base64 overhead for the image tensors:

    [4-byte LE total frame length][4-byte LE header length][JSON header][raw binary tensor payload]

The 4-byte prefix carries the *total* frame length (header + payload) so the
reader knows exactly how many bytes to drain; the second 4-byte field gives the
JSON header length so the header and payload can be split. The JSON header
describes the payload layout::

    {
      "id": "req-0001",
      "type": "infer",                       # "infer" | "ping" | "shutdown"
      "mode": "select_action",               # or "predict_action_chunk"
      "task": "pick up the cup\\n",
      "noise": null,                         # "zeros" forces deterministic sampling
      "tensors": {
        "observation.images.camera1": {"shape": [3, 512, 512], "dtype": "float32", "offset": 0, "nbytes": 3145728},
        ...
      }
    }

``encode_request`` packs the named tensors into the payload in header order;
``decode_request`` unpacks them back. Responses are JSON-only. Control
messages (``ping`` / ``describe`` / ``shutdown``) are JSON-only request frames
built with :func:`encode_control`.

numpy is imported lazily inside the functions that touch payloads, so the
framing layer (and everything importing it, e.g. the CLI) works on a machine
without the inference stack installed. The hot path only pays a ``sys.modules``
lookup per call.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

LEN_BYTES = 4  # width of each length field
PREFIX_BYTES = 8  # total_len + header_len
MAX_HEADER_BYTES = 1 << 20  # 1 MiB
MAX_FRAME_BYTES = 1 << 30  # 1 GiB cap on a single frame
# How long a gateway waits for the worker to answer its ``describe`` before giving up
# (seconds). A worker on an old/slow CPU can spend minutes loading weights -- the fastwam
# text encoder dequantizes ~13GB on the CPU, ~1min on the reference machine and several
# times that on old hardware -- and the wait is cheap: one log line, then silence until the
# worker answers (see ``gateway.WorkerClient``). A backstop for a worker that never comes
# up, not the expected wait; ``tybok serve`` uses the same value for its own socket wait.
WORKER_STARTUP_TIMEOUT = 600.0

# Dtype names the encoder emits and the decoder accepts (the gateway <-> worker contract,
# not numpy's full dtype table); resolved with ``numpy.dtype`` at the use site.
_DTYPE_NAMES = frozenset({"float32", "float64", "int64", "int32", "uint8"})


def _frame(total_len: int, header: bytes, payload: bytes = b"") -> bytes:
    # b"".join over bytes+bytes: CPython's bytes __add__ on a ~6MB payload is
    # an order of magnitude slower than join (~3.5ms vs ~0.2ms), which showed
    # up as ~4ms of gateway CPU per request in encode_request.
    return b"".join((struct.pack("<II", total_len, len(header)), header, payload))


def encode_request(
    request_id: str,
    task: str,
    tensors: dict[str, np.ndarray],
    mode: str = "select_action",
    noise: str | None = None,
    quantize_images: bool = False,
    shm: tuple | None = None,
    rtc: dict | None = None,
) -> bytes | bytearray:
    """Serialize a request: length prefix + JSON header + tensor bytes.

    ``tensors`` maps names (e.g. ``observation.images.camera1``) to numpy
    arrays. ``noise`` optionally forces a deterministic flow-matching noise
    ("zeros") for testing. ``quantize_images`` packs the ``observation.images.*``
    tensors as uint8 (rounding the [0, 1] floats to 0..255; the worker divides
    by 255 back) -- 4x smaller IPC payload at 1/255 pixel precision, opt-in
    because it changes the pixels the model sees. Returns the full frame
    (``bytes``-like, safe for the socket write path and ``decode_request``)
    ready to write to the socket.

    ``shm`` optionally points at a shared-memory region ``(mmap, slot_bytes,
    slot_index)``: the tensor payload is written straight into
    ``slot_index * slot_bytes`` of the region and the returned frame carries
    only the prefix + JSON header (with ``"shm": {"slot": ...}``), so the ~6MB
    payload never traverses the socket / kernel buffers. The caller owns the
    ring-slot lifetime (the reader must have consumed slot N-8 before slot N
    is rewritten). When the payload would not fit the slot, the function
    silently falls back to the inline-payload frame.

    ``rtc`` optionally carries the real-time-chunking block
    (``{"inference_delay": int, "execution_horizon": int | None,
    "prev_chunk_left_over": [[...], ...]}``); it is embedded in the JSON header
    as-is (tiny: <= chunk_size x action_dim floats).

    Either way the frame is packed with exactly one large allocation per call
    (tensors copied in via a writable numpy view): per-tensor ``tobytes()``
    intermediates + ``join`` + concat measures ~4-5ms on a 6.3MB frame while
    this path is ~0.4ms, because the multi-large-allocation pattern drives
    glibc into a slow state.
    """
    import numpy as np

    specs, prep, offset = _tensor_layout(tensors, quantize_images)
    if shm is not None and offset <= shm[1]:
        # shared-memory path: payload into the ring slot, frame carries only
        # the prefix + JSON header (with the slot index); falls back to the
        # inline payload when it would not fit the slot.
        region, slot_bytes, slot = shm
        header_fields = {
            "id": request_id,
            "type": "infer",
            "mode": mode,
            "task": task,
            "noise": noise,
            "quantized_images": quantize_images,
            "tensors": specs,
            "shm": {"slot": slot},
        }
        if rtc is not None:
            header_fields["rtc"] = rtc
        header = json.dumps(header_fields, ensure_ascii=False).encode("utf-8")
        if len(header) > MAX_HEADER_BYTES:
            raise ValueError(f"header too large: {len(header)} bytes")
        dst = np.frombuffer(region, dtype=np.uint8)
        off = slot * slot_bytes
        for arr in prep:
            n = arr.nbytes
            dst[off : off + n] = arr.view(np.uint8).reshape(-1)
            off += n
        return b"".join((struct.pack("<II", PREFIX_BYTES + len(header), len(header)), header))

    header = json.dumps(
        {
            "id": request_id,
            "type": "infer",
            "mode": mode,
            "task": task,
            "noise": noise,
            "quantized_images": quantize_images,
            "tensors": specs,
            "rtc": rtc,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {len(header)} bytes")

    buf = bytearray(PREFIX_BYTES + len(header) + offset)
    struct.pack_into("<II", buf, 0, len(buf), len(header))
    buf[PREFIX_BYTES : PREFIX_BYTES + len(header)] = header
    dst = np.frombuffer(memoryview(buf), dtype=np.uint8)
    off = PREFIX_BYTES + len(header)
    for arr in prep:
        n = arr.nbytes
        dst[off : off + n] = arr.view(np.uint8).reshape(-1)
        off += n
    return buf  # no bytes(buf) copy: a second 6.3MB allocation slows this ~10x


def encode_request_gpu(
    request_id: str,
    task: str,
    tensors: dict[str, np.ndarray],
    mode: str = "select_action",
    noise: str | None = None,
    quantize_images: bool = False,
    device: int = 0,
    rtc: dict | None = None,
) -> tuple[bytes, list]:
    """GPU-direct variant of :func:`encode_request` (see ``--gpu-ipc``).

    The tensor payload is HtoD-copied on the *gateway* side and shared with the
    worker via ``cudaIpcMemHandle`` (``_share_cuda_``); the returned frame is
    header-only and every tensor spec carries a ``gpu`` entry with the import
    metadata (:func:`decode_request` imports it zero-copy). The second return
    value is the list of CUDA tensors the caller must keep alive until the
    worker has imported them -- the gateway holds them in a ring of ``slots``
    requests, mirroring the shm-slot lifetime argument.

    The HtoD copies run on the gateway's current stream; ``_share_cuda_``
    records a cross-process event after them, so the worker's import + copy
    cannot observe a half-written buffer.
    """
    import base64

    import torch

    specs, prep, _ = _tensor_layout(tensors, quantize_images)
    keep: list[torch.Tensor] = []
    for (_name, spec), arr in zip(specs.items(), prep, strict=True):
        t = torch.from_numpy(arr).to(device=device, non_blocking=True)
        (
            dev,
            handle,
            size_bytes,
            offset_bytes,
            ref_h,
            ref_o,
            ev_h,
            ev_sync,
        ) = t.untyped_storage()._share_cuda_()
        spec["gpu"] = {
            "device": dev,
            "handle": base64.b64encode(handle).decode(),
            "size_bytes": size_bytes,
            "offset_bytes": offset_bytes,
            "ref_h": base64.b64encode(ref_h).decode(),
            "ref_o": ref_o,
            "ev_h": base64.b64encode(ev_h).decode(),
            "ev_sync": bool(ev_sync),
            "shape": list(arr.shape),
        }
        keep.append(t)
    header = json.dumps(
        {
            "id": request_id,
            "type": "infer",
            "mode": mode,
            "task": task,
            "noise": noise,
            "quantized_images": quantize_images,
            "tensors": specs,
            "rtc": rtc,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {len(header)} bytes")
    return b"".join((struct.pack("<II", PREFIX_BYTES + len(header), len(header)), header)), keep


def _tensor_layout(
    tensors: dict[str, np.ndarray], quantize_images: bool
) -> tuple[dict[str, dict[str, Any]], list[np.ndarray], int]:
    """Shared encode layout pass: quantize (opt-in), spec dicts, prep arrays."""
    import numpy as np

    specs: dict[str, dict[str, Any]] = {}
    prep: list[np.ndarray] = []
    offset = 0
    for name, arr in tensors.items():
        if quantize_images and name.startswith("observation.images."):
            arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        dtype = str(arr.dtype)
        if dtype not in _DTYPE_NAMES:
            raise ValueError(f"unsupported dtype {dtype} for tensor {name}")
        nbytes = arr.nbytes
        specs[name] = {"shape": list(arr.shape), "dtype": dtype, "offset": offset, "nbytes": nbytes}
        prep.append(arr)
        offset += nbytes
    return specs, prep, offset


def decode_request(
    frame: bytes | bytearray, shm: tuple | None = None
) -> tuple[str, str, str, dict[str, np.ndarray], str | None, bool]:
    """Unpack a request frame -> (request_id, mode, task, tensors, noise, quantized_images).

    ``quantized_images`` is True when the gateway packed the ``observation.images.*``
    tensors as uint8 (the caller divides by 255 before feeding the model).

    Zero-copy: tensors are read straight out of the frame payload via
    ``np.frombuffer`` offsets (no payload slice / no defensive copy). When the
    frame header carries ``"shm"`` (see :func:`encode_request`), ``shm`` must
    be the ``(mmap, slot_bytes)`` tuple of the shared region and the tensors
    are read from the region slot instead; the returned views keep the region
    alive, so the slot must not be rewritten until the caller is done.
    """
    import numpy as np

    if len(frame) < PREFIX_BYTES:
        raise ValueError("frame too short")
    total_len, header_len = struct.unpack("<II", frame[:PREFIX_BYTES])
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {header_len} bytes")
    if total_len != len(frame):
        raise ValueError(f"frame length mismatch: header says {total_len}, got {len(frame)}")
    header = json.loads(frame[PREFIX_BYTES : PREFIX_BYTES + header_len].decode("utf-8"))

    shm_slot = header.get("shm")
    if shm_slot is not None:
        if shm is None:
            raise ValueError("frame requires the shared-memory region (worker started without --shm-ipc?)")
        region, slot_bytes = shm
        base = int(shm_slot["slot"]) * slot_bytes
        payload = region
    else:
        base = PREFIX_BYTES + header_len
        payload = frame

    tensors: dict[str, np.ndarray] = {}
    for name, spec in header.get("tensors", {}).items():
        if spec["dtype"] not in _DTYPE_NAMES:
            raise ValueError(f"unsupported dtype {spec['dtype']} for tensor {name}")
        dtype = np.dtype(spec["dtype"])
        gpu = spec.get("gpu")
        if gpu is not None:
            tensors[name] = _import_gpu_tensor(gpu, spec["dtype"], spec["shape"])
            continue
        start = spec["offset"]
        # Zero-copy view into the frame payload / shm slot: the caller (worker
        # ``handle_frame``) consumes the tensors synchronously while the buffer
        # is alive, so no defensive copy is needed (saves one ~6MB memcpy per
        # request). The payload layout is contiguous by construction.
        arr = np.frombuffer(payload, dtype=dtype, count=int(np.prod(spec["shape"])), offset=base + start)
        tensors[name] = arr.reshape(spec["shape"])
    return (
        header.get("id", ""),
        header.get("mode", "select_action"),
        header.get("task", ""),
        tensors,
        header.get("noise"),
        bool(header.get("quantized_images", False)),
    )


def _import_gpu_tensor(gpu: dict, dtype_name: str, shape: list) -> object:
    """Import a gateway-shared CUDA tensor zero-copy (see --gpu-ipc).

    Mirrors ``torch.multiprocessing.reductions.rebuild_cuda_tensor``: open the
    ``cudaIpcMemHandle`` with ``_new_shared_cuda`` (which also waits on the
    producer's cross-process event, so the HtoD writes are visible) and wrap it
    as a contiguous tensor of the requested shape/dtype. The returned tensor
    keeps the mapped storage alive; the mapping is closed when it is destroyed.
    """
    import base64

    import torch

    if not torch.cuda.is_available():
        raise ValueError("frame carries GPU tensors but CUDA is unavailable")
    torch.cuda._lazy_init()  # required before opening a handle (as in reductions.rebuild_cuda_tensor)
    storage = torch.UntypedStorage._new_shared_cuda(
        gpu["device"],
        base64.b64decode(gpu["handle"]),
        gpu["size_bytes"],
        gpu["offset_bytes"],
        base64.b64decode(gpu["ref_h"]),
        gpu["ref_o"],
        base64.b64decode(gpu["ev_h"]),
        bool(gpu["ev_sync"]),
    )
    dtype = getattr(torch, dtype_name)
    ts = torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)
    # contiguous row-major strides for `shape` (the gateway tensors are contiguous)
    strides = [1] * len(shape)
    numel = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = numel
        numel *= shape[i]
    return torch._utils._rebuild_tensor(ts, 0, list(shape), strides)


def decode_rtc(frame: bytes | bytearray) -> dict | None:
    """Extract the optional ``"rtc"`` block from an infer request header.

    The block is JSON-only (small, unlike the tensor payload), so the worker
    re-parses the header instead of threading it through ``decode_request``'s
    return tuple (which the tests unpack positionally).
    """
    if len(frame) < PREFIX_BYTES:
        return None
    _, header_len = struct.unpack("<II", frame[:PREFIX_BYTES])
    if header_len > MAX_HEADER_BYTES:
        return None
    try:
        header = json.loads(frame[PREFIX_BYTES : PREFIX_BYTES + header_len].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    rtc = header.get("rtc")
    return rtc if isinstance(rtc, dict) else None


def encode_control(request_id: str, mtype: str) -> bytes:
    """JSON-only control request frame: ``ping`` | ``describe`` | ``shutdown``."""
    header = json.dumps({"id": request_id, "type": mtype}, ensure_ascii=False).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {len(header)} bytes")
    return _frame(PREFIX_BYTES + len(header), header)


def encode_response(request_id: str, ok: bool, data: dict[str, Any], error: str | None = None) -> bytes:
    header = json.dumps({"id": request_id, "ok": ok, "data": data, "error": error}, ensure_ascii=False).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {len(header)} bytes")
    return _frame(PREFIX_BYTES + len(header), header)


def decode_response(frame: bytes) -> dict[str, Any]:
    if len(frame) < PREFIX_BYTES:
        raise ValueError("frame too short")
    _, header_len = struct.unpack("<II", frame[:PREFIX_BYTES])
    header = json.loads(frame[PREFIX_BYTES : PREFIX_BYTES + header_len].decode("utf-8"))
    return header
