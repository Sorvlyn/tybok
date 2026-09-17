"""Image decoding and preprocessing for the WebSocket gateway.

The gateway receives camera frames as raw bytes (JPEG/PNG) or base64 strings,
decodes them with Pillow and converts them to the same ``(C, H, W)`` float32
tensors in ``[0, 1]`` that the reference demo feeds into the policy. Resizing
with left/top padding to the model's target resolution is done here (plan 2:
the gateway owns image processing), so the inference worker only sees
model-ready tensors.
"""

from __future__ import annotations

import base64
import io
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
    """Resize a (b, c, h, w) image without distortion, padding on LEFT and TOP.

    This is the smolvla/openpi convention (the checkpoint's ``resize_imgs_with_
    padding`` target is stored as (width, height)). ``pad_value`` is keyword-only
    on purpose: callers historically used different values (0, -1).
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")
    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def resize_with_pad_center(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
    """Resize a (b, c, h, w) image without distortion, CENTERED padding.

    This is the pi0.5/openpi convention (``resize_with_pad_torch``): the extra
    pad pixel goes to the bottom/right (``divmod``), unlike the smolvla
    left/top padding above.
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")
    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_img = F.pad(resized_img, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_value)
    return padded_img


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/... bytes to an RGB uint8 numpy array (H, W, 3)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        return np.asarray(img)


def image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _decode_with_torchvision(data: bytes) -> torch.Tensor:
    """torchvision decoder (libjpeg-turbo): direct ``(3, H, W)`` uint8 output,
    so the gateway hot path skips the HWC->CHW copy and the numpy round-trip.
    Roughly 1.3-1.8x faster than the Pillow path on 512x512 JPEG/PNG.
    """
    import torchvision.io as torchvision_io  # noqa: PLC0415

    # ``frombuffer`` warns on the immutable bytes aiohttp hands us; the bytearray copy is a memcpy
    # of the compressed frame, next to a full JPEG decode.
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if data[:3] == b"\xff\xd8\xff":
        img = torchvision_io.decode_jpeg(buf)
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        img = torchvision_io.decode_png(buf)
    else:
        raise ValueError("unsupported image format (expected JPEG or PNG)")
    # ``decode_*`` also has a batch overload returning ``list[Tensor]``; this path passes a single
    # buffer, so the result is one ``(3, H, W)`` tensor.
    return cast(torch.Tensor, img)


def decode_image_to_tensor(data: bytes) -> torch.Tensor:
    """Decode JPEG/PNG bytes to a ``(3, H, W)`` float32 tensor in ``[0, 1]``.

    Prefers the torchvision decoder (libjpeg-turbo, CHW output, no transpose)
    and falls back to the Pillow path when torchvision is unavailable or the
    format is unsupported.
    """
    try:
        img = _decode_with_torchvision(data)
    except Exception:  # noqa: BLE001 - Pillow handles more formats
        return image_to_tensor(decode_image_bytes(data))
    return img.to(torch.float32) / 255.0


def base64_to_tensor(b64: str) -> torch.Tensor:
    return decode_image_to_tensor(base64.b64decode(b64))


def resize_stretch(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Stretch-resize a (b, c, h, w) image directly to (height, width).

    fastWAM convention: no aspect-preserving pad (the model owns the input
    resolution and resizes every camera to its per-camera target)."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")
    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img
    return F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)


def prepare_image_tensor(img: torch.Tensor, target: tuple[int, int], pad_mode: str = "top-left") -> torch.Tensor:
    """Resize/pad a (C, H, W) float tensor to the model target (height, width).

    ``target`` follows the checkpoint convention ``(width, height)``. ``pad_mode``
    selects the transform: ``"top-left"`` (smolvla reference convention,
    default), ``"center"`` (pi0.5 openpi convention) or ``"stretch"``
    (fastWAM: direct resize, no padding).
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if pad_mode == "center":
        out = resize_with_pad_center(img, target[1], target[0], pad_value=0)
    elif pad_mode == "stretch":
        out = resize_stretch(img, target[1], target[0])
    else:
        out = resize_with_pad(img, target[1], target[0], pad_value=0)
    return out.squeeze(0)
