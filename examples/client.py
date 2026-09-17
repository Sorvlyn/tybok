"""Example WebSocket client for the TyBoK gateway (moved out of the server-side
``tybok`` package into ``examples/``).

Usage (run from the project root, i.e. the ``tybok/`` directory)::

    python examples/client.py --url ws://127.0.0.1:8765/ws \\
        --task "pick up the cup" --state 0,0,0,0,0,0,0,0 \\
        --image camera1=frame1.jpg,camera3=frame3.jpg

Or replay a reference frame dump (``frame.pt``; needs torch/PIL)::

    python examples/client.py --from-ref <frame.pt> --noise-zero --frames 3

Non-Python clients (websocat / wscat / curl) are described in README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json

import aiohttp


async def send_frame(url: str, payload: dict, timeout: float = 120.0) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, timeout=timeout, max_msg_size=0) as ws:
            await ws.send_json(payload)
            resp = await asyncio.wait_for(ws.receive(), timeout=timeout)
            return json.loads(resp.data)


def image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def tensor_to_png_base64(img) -> str:
    """(C,H,W) float32 in [0,1] -> base64 PNG string."""
    import numpy as np
    from PIL import Image

    arr = (img.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_images(spec: str, parser: argparse.ArgumentParser) -> dict[str, str]:
    """``CAM=FILE[,CAM=FILE...]`` -> ``{camera: base64 image}`` (see ``--image``).

    One comma-separated value rather than a repeatable flag, matching the server-side
    ``--camera-alias``: repeating the flag would silently keep only the last value in
    most shells.
    """
    images: dict[str, str] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        cam, sep, path = item.partition("=")
        if not sep or not cam.strip() or not path.strip():
            parser.error(f"--image expects CAM=FILE, got {item!r}")
        images[cam.strip()] = image_file_to_base64(path.strip())
    return images


def build_payload_from_ref(frame_path: str) -> dict:
    import torch

    frame = torch.load(frame_path, weights_only=False)
    images = {}
    for key, val in frame.items():
        if key.startswith("observation.images."):
            images[key[len("observation.images.") :]] = tensor_to_png_base64(val)
    state = (
        frame["observation.state"].tolist()
        if hasattr(frame["observation.state"], "tolist")
        else list(frame["observation.state"])
    )
    return {"images": images, "state": state, "task": frame.get("task", ""), "mode": "select_action"}


def _build_parser() -> argparse.ArgumentParser:
    """The example client's command line (the module docstring shows the common forms)."""
    parser = argparse.ArgumentParser(description="TyBoK WebSocket client (examples/client.py)")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--task", default="pick up the cup")
    parser.add_argument("--state", default="", help="comma-separated floats")
    parser.add_argument(
        "--image",
        default="",
        metavar="CAM=FILE[,CAM=FILE...]",
        help="camera images, one comma-separated value (not repeatable -- like the "
        "server-side --camera-alias, a repeated flag would silently keep only its last "
        "value in most shells): --image camera1=frame1.jpg,camera3=frame3.jpg",
    )
    parser.add_argument("--chunk", action="store_true", help="request the full action chunk")
    parser.add_argument("--from-ref", default=None, help="build the request from a frame.pt dump")
    parser.add_argument("--noise-zero", action="store_true", help="use deterministic zero noise (testing)")
    parser.add_argument("--frames", type=int, default=1, help="number of frames to send (per-step mode)")
    # Real-Time Chunking (the worker must be started with --rtc).
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="send an rtc block with each request (real-time chunking; the worker "
        "must be started with --rtc). Implies --chunk",
    )
    parser.add_argument(
        "--rtc-prev",
        default=None,
        help="prev_chunk_left_over as JSON, e.g. '[[0.1,0.2,...],[0.3,0.4,...]]' "
        "(model-space normalized actions from the previous chunk's unexecuted tail)",
    )
    parser.add_argument("--rtc-delay", type=int, default=0, help="inference_delay in steps (default 0)")
    parser.add_argument(
        "--rtc-horizon", type=int, default=None, help="execution_horizon in steps (default: engine config)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.from_ref:
        payload = build_payload_from_ref(args.from_ref)
    else:
        images = parse_images(args.image, parser)
        state = [float(x) for x in args.state.split(",") if x != ""]
        payload = {
            "images": images,
            "state": state,
            "task": args.task,
            "mode": "predict_action_chunk" if args.chunk else "select_action",
        }

    if args.noise_zero:
        payload["noise"] = "zeros"
    if args.rtc:
        payload["mode"] = "predict_action_chunk"
        rtc_block: dict = {"inference_delay": args.rtc_delay}
        if args.rtc_prev:
            rtc_block["prev_chunk_left_over"] = json.loads(args.rtc_prev)
        if args.rtc_horizon is not None:
            rtc_block["execution_horizon"] = args.rtc_horizon
        payload["rtc"] = rtc_block
    mode = payload.get("mode", "select_action")
    for i in range(args.frames):
        resp = asyncio.run(send_frame(args.url, payload))
        if resp.get("ok"):
            action = resp["action"]
            if mode == "predict_action_chunk":
                print(f"[{i}] chunk shape={resp.get('shape')} first={action[0] if action else None}")
                if "action_normalized" in resp:
                    norm = resp["action_normalized"]
                    print(f"[{i}]   action_normalized shape={resp.get('shape')} first={norm[0] if norm else None}")
            else:
                print(f"[{i}] action={action}")
        else:
            print(f"[{i}] ERROR: {resp.get('error')}")
            if not resp.get("ok"):
                break


if __name__ == "__main__":
    main()
