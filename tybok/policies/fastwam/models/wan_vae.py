# Derived from diffusers <https://github.com/huggingface/diffusers>. Copyright 2025 The Wan Team
# and The HuggingFace Team. Licensed under the Apache License, Version 2.0; modified for
# TyBoK in 2026. See the NOTICE file.

"""Wan2.2 3D VAE encoder (single-frame encode).

Port of the *encoder* half of ``diffusers.AutoencoderKLWan``:

- spatial compression x16 (patch_size 2 patchify + 3 stride-2 conv downsamples)
- 48 latent channels with diffusers standardization
  ``(mu - latents_mean) / latents_std`` applied in fp32 (diffusers returns raw
  latents)
- deterministic posterior: encoder emits 2*z_dim channels, ``mode()`` keeps the
  first z_dim (mean half)
- encode runs the temporal dimension with causal convs; a single input frame
  (T=1) goes through one chunk and never triggers the temporal stride convs
  (the diffusers feature-cache path skips ``time_conv`` on the first chunk)

Only the encoder + ``quant_conv`` state is loaded (keys ``encoder.*`` /
``quant_conv.*``); the decoder is not needed for action inference.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as functional

# Shared stateless SiLU; ``nn.SiLU`` exists in every torch this package supports (torch>=2.7.1), so
# there is no fallback to keep alive (the old ``except`` bound ``silu = None``, i.e. a callable that
# would have crashed at the first use rather than degraded).
silu = nn.SiLU()


def _to_2d(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """[B, C, T, H, W] -> [B*T, C, H, W] (+ dims for the restore)."""
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), b, t


def _from_2d(x: torch.Tensor, b: int, t: int, c: int) -> torch.Tensor:
    _, _, h, w = x.shape
    return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class WanCausalConv3d(nn.Conv3d):
    """Conv3d with causal temporal padding (pad left 2*pad_t, right 0)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # torch normalises an int / 3-tuple padding to a 3-tuple of ints; the 'same' / 'valid'
        # strings would break the causal pad below, so they are rejected here instead.
        if isinstance(self.padding, str):
            raise ValueError("WanCausalConv3d does not support 'same' / 'valid' padding")
        pad_t, pad_h, pad_w = self.padding
        self._temporal_pad = (pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0)
        self.padding = (0, 0, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded = functional.pad(input, self._temporal_pad)
        return super().forward(padded)


class WanRMSNorm(nn.Module):
    """RMS norm over the channel dim for [B, C, H, W] (images=True) or
    [B, C, T, H, W] (images=False) tensors. ``gamma`` shape follows diffusers
    WanRMS_norm: (dim, 1, 1) for 2D data, (dim, 1, 1, 1) for 3D data."""

    def __init__(self, dim: int, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        self.images = images
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcastable_dims))
        self.bias = nn.Parameter(torch.zeros(dim, *broadcastable_dims)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
        normalized = functional.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class WanResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = WanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim, images=False)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.conv1(silu(self.norm1(x)))
        x = self.conv2(silu(self.norm2(x)))
        return x + h


class WanAttentionBlock2D(nn.Module):
    """Single-head 2D self attention used by the VAE mid block."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = WanRMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = functional.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class WanMidBlock(nn.Module):
    """Residual + attention + residual at the bottom of the encoder."""

    def __init__(self, dim: int):
        super().__init__()
        self.resnets = nn.ModuleList([WanResidualBlock(dim, dim)])
        self.attentions = nn.ModuleList([WanAttentionBlock2D(dim)])
        self.resnets.append(WanResidualBlock(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = attn(x)
            x = resnet(x)
        return x


class _WanSpatialDownsample(nn.Module):
    """2D stride-2 downsampling (mirrors WanResample's ``resample`` for the
    first chunk; the temporal stride conv is never triggered for T=1)."""

    def __init__(self, dim: int, with_time_conv: bool):
        super().__init__()
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        # module kept for weight completeness (never executed on T=1 chunks)
        self.time_conv = (
            WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)) if with_time_conv else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x2d, batch, time = _to_2d(x)
        x2d = self.resample(x2d)
        return _from_2d(x2d, batch, time, c)


class WanResidualDownBlock(nn.Module):
    """Encoder stage: resnets + (optional) spatial downsample; AvgDown shortcut."""

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temperal_downsample: bool, down_flag: bool):
        super().__init__()
        self.avg_shortcut = _AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            down_flag=down_flag,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = _WanSpatialDownsample(out_dim, temperal_downsample) if down_flag else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)


class _AvgDown3D(nn.Module):
    """Grouped mean-pool shortcut of the encoder down blocks (no parameters)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, down_flag: bool):
        super().__init__()
        factor_s = 2 if down_flag else 1
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = functional.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        factor_t, factor_s = self.factor_t, self.factor_s
        x = x.view(b, c, t // factor_t, factor_t, h // factor_s, factor_s, w // factor_s, factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // factor_t, h // factor_s, w // factor_s)
        x = x.view(b, -1, self.group_size, t // factor_t, h // factor_s, w // factor_s)
        return x.mean(dim=2)


class WanVAEEncoder(nn.Module):
    """Encoder + quant_conv (+ latent standardization), T=1 only."""

    def __init__(
        self,
        base_dim: int = 160,
        z_dim: int = 48,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] | None = None,
        in_channels: int = 12,
        patch_size: int = 2,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [False, True, True]
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.upsampling_factor = 16
        self.temporal_downsample_factor = 4
        dims = [base_dim * u for u in [1] + dim_mult]

        self.encoder = _WanEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dims=dims,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_downsample=temperal_downsample,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)

        mean = latents_mean if latents_mean is not None else [0.0] * z_dim
        std = latents_std if latents_std is not None else [1.0] * z_dim
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32).view(1, z_dim, 1, 1, 1),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel-space [B,3,T,H,W] -> patched [B, 3*ps*ps, T, H/ps, W/ps]."""
        if self.patch_size == 1 or x.dim() != 5:
            return x
        b, c, t, h, w = x.shape
        ps = self.patch_size
        x = x.view(b, c, t, h // ps, ps, w // ps, ps)
        x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
        return x.view(b, c * ps * ps, t, h // ps, w // ps)

    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode one image frame into standardized latents.

        Args:
            frame: [1, 3, 1, H, W] (or [1, 3, H, W]) in [-1, 1], model dtype
                   (bf16). H/W must be multiples of 16.

        Returns:
            fp32 [1, z_dim, 1, H/16, W/16] standardized latent (posterior mean).
        """
        if frame.ndim == 4:
            frame = frame.unsqueeze(2)
        if frame.ndim != 5 or frame.shape[0] != 1 or frame.shape[1] != 3:
            raise ValueError(f"expected a single [1,3,1,H,W] frame, got {tuple(frame.shape)}")
        if frame.shape[2] != 1:
            raise ValueError(f"fastWAM deploy encodes a single frame (T=1); got T={frame.shape[2]}")
        dtype = self.quant_conv.weight.dtype
        x = frame.to(dtype=dtype)
        x = self.patchify(x)
        x = self.encoder(x)
        x = self.quant_conv(x)  # [1, 2*z_dim, 1, h, w]
        latent_mode = x[:, : self.z_dim]  # DiagonalGaussianDistribution.mode()
        latent_mode = latent_mode.float()
        latents_mean = self.latents_mean.float().to(latent_mode.device)
        latents_std = self.latents_std.float().to(latent_mode.device)
        return (latent_mode - latents_mean) / latents_std


class _WanEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        z_dim: int,
        dims: list[int],
        dim_mult: list[int],
        num_res_blocks: int,
        temperal_downsample: list[bool],
    ):
        super().__init__()
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            last = i == len(dim_mult) - 1
            self.down_blocks.append(
                WanResidualDownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_downsample=bool(temperal_downsample[i]) if not last else False,
                    down_flag=not last,
                )
            )
        self.mid_block = WanMidBlock(dims[-1])
        self.norm_out = WanRMSNorm(dims[-1], images=False)
        self.conv_out = WanCausalConv3d(dims[-1], z_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = silu(self.norm_out(x))
        return self.conv_out(x)


def load_wan_vae_encoder(vae_dir: str, dtype: torch.dtype = torch.bfloat16) -> WanVAEEncoder:
    """Build + load the VAE encoder from a diffusers ``vae/`` directory."""
    cfg_path = os.path.join(vae_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise ValueError(f"no config.json in vae dir {vae_dir!r}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = WanVAEEncoder(
        base_dim=int(cfg.get("base_dim", 160)),
        z_dim=int(cfg.get("z_dim", 48)),
        dim_mult=[int(v) for v in cfg.get("dim_mult", [1, 2, 4, 4])],
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        temperal_downsample=[bool(v) for v in cfg.get("temperal_downsample", [False, True, True])],
        in_channels=int(cfg.get("in_channels", 12)),
        patch_size=int(cfg.get("patch_size", 2)),
        latents_mean=[float(v) for v in cfg.get("latents_mean", [])],
        latents_std=[float(v) for v in cfg.get("latents_std", [])],
    )
    # load every encoder./quant_conv tensor (diffusers names, decoder skipped)
    import glob

    params = {name: p for name, p in model.named_parameters()}
    loaded = set()
    for path in sorted(glob.glob(os.path.join(vae_dir, "*.safetensors"))):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in params:
                    continue  # decoder.* / post_quant_conv.*
                t = f.get_tensor(key)
                params[key].data.copy_(t.reshape(params[key].shape).to(dtype=dtype))
                loaded.add(key)
    missing = [name for name in params if name not in loaded]
    if missing:
        raise ValueError(f"vae encoder weights missing from checkpoint: {missing[:8]} ...")
    model.to(dtype=dtype)
    return model
