"""fastWAM deployment engine (registered as ``model_type="fastwam"``).

Batch-1 inference backend for the Wan2.2-MoT-based FastWAM policy: UMT5 text
encoder, Wan VAE encoder, MoT video prefill + action Euler denoise, and MIN_MAX
pre/post normalization. :class:`FastWAMEngine` is the deployment entry point.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch

from ...base import PolicyEngine, compile_region
from ...registry import register, reject_unsupported_flags
from .config import FastWAMConfig
from .models.dit_block import pack_attention_qkv, pack_attention_qkv_fp8
from .models.fastwam import FastWAM
from .models.fp8_linear import FP8Linear, fp8ify_structural
from .models.policy import FastWAMPolicy, fp8_checkpoint_files, load_model
from .models.umt5 import load_umt5
from .models.wan_vae import load_wan_vae_encoder
from .preprocess import PostProcessor, PreProcessor, load_normalizer, load_unnormalizer
from .tokenizer import UMT5Tokenizer

log = logging.getLogger("tybok.fastwam")


@register("fastwam")
class FastWAMEngine(PolicyEngine):
    # Flags this backend does not implement: pi05/smolvla-owned switches plus
    # removed legacy fastwam flags. The worker forwards model-agnostic kwargs with
    # these inert defaults; a differing value means the flag was requested and is
    # rejected by ``registry.reject_unsupported_flags`` -- listed (not deleted) so
    # old callers get an explicit error instead of a silent eager fallback.
    # ``compile_model`` / ``graph`` are *named* params, so they are not in this table.
    _UNSUPPORTED_FLAGS = {
        # pi05 / smolvla Triton tiers
        "tl_fused_vit": False,
        "tl_llm_flash_attn": False,
        "tl_llm_fused_attn": False,
        "tl_vit_oproj": False,
        "tl_fused_expert": False,
        "tl_fp8_llm_mlp": False,
        "tl_fp8_expert_mlp": False,
        "vit_mlp_dtype": None,
        "skip_empty_images": False,
        "pad_free": False,
        # removed legacy fastwam flags
        "fast": False,
        "act_qkv_sdpa_attn": False,
        "vision_fp16_qkv": False,
        "vision_triton_qkv": False,
        "prefill_sdpa_attn": False,
        "triton_txt_attn": False,
        "triton_txt_ffn": False,
        "fused_text_encoder": False,
        "shm_ipc": False,
    }

    # Flag table: the aligned trailing comments are the index of this signature. The signature
    # carries ``# fmt: skip`` so the formatter leaves the table alone; the body is formatted.
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "auto",
        compile_model: bool = False,
        warmup: bool = True,
        graph: bool = False,
        graph_cameras: tuple[int, ...] | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        sampler: str = "euler",
        # Accepted for CLI uniformity but not applicable: both paths always use the video
        # prefill KV cache (unconditional upstream too), so ``False`` is rejected.
        cache_expert_prefix_kv: bool = True,
        cache_prompt: bool = True,  # memoize the task's UMT5 embedding (--no-prompt-cache disables)
        profile: bool = False,  # one-shot startup latency profile
        overlap: bool = True,  # fuse step 0 into the video prefill (CUDA; two-stream, bit-exact)
        tokenizer_dir: str | None = None,
        text_encoder_dir: str | None = None,
        vae_dir: str | None = None,
        text_encoder_device: str = "cpu",
        text_emb_cpu: bool = True,
        video_fp8: bool = True,   # all-fp8 tier; only honored when fp8 files are present
        action_fp8: bool = True,  # disable with --video-bf16 / --action-bf16 / --text-bf16
        text_fp8: bool = True,
        action_fused: bool = False,  # fused self/cross-attn + FFN kernels (needs fp8-resident)
        action_fused_split: bool = False,  # non-cooperative form: plain launch per phase
        action_pre_fused: bool = False,  # fused time path + context kv (needs --cu-fused-adit)
        video_pre_fused: bool = False,  # resident table/mask cache + exact time embedding (bitwise)
        video_fused: bool = False,   # prefill block tail (norm2+FFN+gate) uses one fused kernel
        video_fused_split: bool = False,  # non-cooperative fallback: plain launch per phase
                                          # (no full-GPU co-residency required)
        camera_alias: dict[str, str] | None = None,
        action_context_cache: bool = True,
        pack_qkv: bool = False,
        text_fused: bool = False,   # one fused kernel per UMT5 attn/FFN sublayer
                                    # (requires fp8-resident + sm_89)
        text_fused_split: bool = False,  # non-cooperative form: plain launch per phase
        require_fused: bool = False,   # hard error instead of degrading (default: graceful)
        **kwargs: Any,
    ):  # fmt: skip
        super().__init__(checkpoint_dir, device=device)
        # Re-check here: ``create_engine`` already validates the raw kwargs, but
        # ``**kwargs`` backends can also be built directly (idempotent; raises on the
        # first offender).
        reject_unsupported_flags(type(self), "fastwam", kwargs)
        # ``--compile`` and the fused CUDA kernels are mutually exclusive: inside a
        # compiled region dynamo can only break the graph around the opaque kernel calls,
        # so the fusion is lost and the compile cost buys nothing -> explicit error
        # instead of silent degradation.
        fused_flags = [
            name
            for name, on in (
                ("--pack-qkv", pack_qkv),
                ("--cu-fused-adit", action_fused or action_fused_split),
                ("--cu-fused-vdit", video_fused or video_fused_split),
                ("--cu-fused-text-encoder", text_fused or text_fused_split),
                ("--action-pre-fused", action_pre_fused),
                ("--video-pre-fused", video_pre_fused),
            )
            if on
        ]
        if compile_model and fused_flags:
            raise NotImplementedError(
                "--compile is not supported together with the fused CUDA kernels "
                f"({', '.join(fused_flags)}); drop --compile or the fused flag"
            )
        # ``--compile`` compiles the DiT core's two hot regions, not ``_denoise_core`` as a
        # whole (see ``_install_compile``); installed after the eager warmup, so here it is
        # only a flag.
        self.compile_model = bool(compile_model)
        # ``--graph``: capture the DiT core (video prefill + action denoise) into a CUDA
        # graph. Every tensor the captured graph reads must be held by the runner
        # (``keepalive``) -- a local closure reference is not enough, it can be freed
        # before replay. The graph mirrors whichever eager branch is selected (overlap
        # on/off), so resolve ``overlap`` first. Streams/events must also pre-date capture.
        self.graph_requested = bool(graph)
        self.graph_enabled = self.graph_requested and str(self.device).startswith("cuda")
        self._graph_runner = None
        if graph_cameras:
            # Accepted for CLI uniformity but a no-op for fastwam: the image slots are
            # concatenated into one frame *outside* the captured core, so a single graph
            # covers every camera count.
            log.info(
                f"--graph-cameras {tuple(graph_cameras)}: no-op for fastwam (one graph "
                f"covers every camera count; the image slots are concatenated into a "
                f"single frame before the graph)"
            )
        if sampler not in ("euler",):
            raise ValueError(f"sampler {sampler!r} not supported for fastwam yet (only 'euler' matches the reference)")
        if not cache_expert_prefix_kv:
            # Both paths always use the video KV cache, so a no-op would mislead.
            raise NotImplementedError(
                "--no-expert-prefix-kv-cache is not applicable to the fastwam backend "
                "(the video prefill KV cache is unconditional)"
            )

        # ``overlap``: fuse the first denoise step into the video prefill (layer i on a
        # side stream gated on that layer's KV). Bit-exact with the sequential path; the
        # cooperative block kernels cannot co-run, so the gain is ~0 on a saturated GPU.
        #
        # ``--compile`` turns it off: the overlap branch is a separate multi-stream entry
        # that cannot be captured together with a compiled core
        # (``cudaErrorStreamCaptureUnjoined``).
        self.overlap = bool(overlap) and str(self.device).startswith("cuda") and not self.compile_model
        if self.compile_model and bool(overlap):
            log.info(
                "--compile: overlap is incompatible (the compiled path is the sequential "
                "prefill->denoise core; the overlap branch is the separate "
                "_prefill_video_layer_action_layer_overlap entry and its multi-stream glue cannot be "
                "captured together with a compiled core); "
                "overlap auto-disabled"
            )

        self._action_fused_runner = None
        self._action_pre_runner = None
        self._video_pre_runner = None
        self._video_fused_runner = None
        self._video_fused_attn_runner = None
        self._text_fused_runner = None

        self.config = FastWAMConfig.from_pretrained(
            checkpoint_dir,
            tokenizer_dir=tokenizer_dir,
            text_encoder_dir=text_encoder_dir,
            vae_dir=vae_dir,
        )
        if num_steps is not None:
            self.config.num_inference_steps = int(num_steps)
        if seed is not None:
            self.config.inference_seed = seed
        # The checkpoint config declares the image slots the model consumes; for fastWAM
        # they are concatenated width-wise into ONE frame before the policy, so report both.
        log.info(
            f"image slots from the checkpoint config: {self.config.image_feature_keys} "
            f"(all slots are concatenated into one "
            f"{self.config.image_size[0]}x{self.config.image_size[1]} frame)"
        )

        if self.device.startswith("cuda"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # ------------------------------------------------------------------ #
        # build frozen side components (text encoder may sit on CPU, prompt-only)
        # ------------------------------------------------------------------ #
        t0 = time.perf_counter()
        te_dir = self.config.text_encoder_dir
        log.info(f"loading UMT5 text encoder from {te_dir} ...")
        te_on_gpu = text_encoder_device in ("cuda", "auto")
        # Non-cooperative form (default): a plain launch per phase, no full-GPU
        # co-residency. ``*_fused_split`` implies the matching ``*_fused``.
        if action_fused_split:
            action_fused = True
        if text_fused_split:
            text_fused = True
        if video_fused_split:
            video_fused = True

        # fp8-resident direct load needs a CUDA device *and* fp8 files in the dir
        # (no files -> always bf16, no silent requant). The fused text tier needs fp8.
        te_has_fp8 = bool(fp8_checkpoint_files(te_dir))
        text_resident = te_on_gpu and text_fp8 and te_has_fp8
        if text_fused:
            blockers = []
            if not self.device.startswith("cuda"):
                blockers.append("CUDA device required (fused kernels are hand-written CUDA)")
            else:
                _cap = tuple(torch.cuda.get_device_capability(self.device))
                if _cap < (8, 9):
                    blockers.append(f"requires sm_89+ (fp8 e4m3 mma), this machine is sm_{_cap[0]}{_cap[1]}")
            if not text_resident:
                blockers.append("UMT5 is not fp8-resident (requires --text-encoder-device cuda + fp8 files)")
            if self._fused_or_eager("text encoder fusion", blockers, require_fused):
                # Nothing is packed yet, so degrading here is still safe.
                text_fused_split = self._resolve_split("text-encoder", text_fused_split, require_fused)
            else:
                text_fused = False
        self.text_encoder = load_umt5(te_dir, dtype=torch_dtype, fp8_resident=text_resident)
        if te_on_gpu:
            n_fp8 = sum(1 for m in self.text_encoder.modules() if isinstance(m, FP8Linear))
            self.text_encoder.to(self.device)
            # embedding table stays on CPU unless --no-text-emb-cpu
            if text_emb_cpu:
                self.text_encoder.encoder.embed_tokens.to("cpu")
            if text_fused:
                # Packing is destructive (frees the original q/k/v and wi_0/wi_1 weights),
                # so after this the fused kernels are the only path; install the hooks
                # before running any forward.
                self.text_encoder.encoder.pack_fused(
                    self.config.context_len,
                    attn=True,
                    ffn=True,
                )
                log.info("text encoder fused (attn+ffn) fp8")

                from .models.fused_text import FusedAttnRunner, FusedTextRunner

                _t0 = time.perf_counter()
                _stack = self.text_encoder.encoder
                # kernel geometry is fixed at compile time; a mismatch is an explicit error
                _te_blk = _stack.block[0].layer[1]
                _dense = _te_blk.DenseReluDense
                if tuple(_dense.wi_weight.shape) != (20480, 4096) or _dense.wo.weight.shape != (4096, 10240):
                    raise NotImplementedError(
                        "--cu-fused-text-encoder kernels have fixed UMT5-XXL geometry at compile time (4096->2x10240->4096), "
                        f"this checkpoint is {tuple(_dense.wi_weight.shape)} / "
                        f"{tuple(_dense.wo.weight.shape)}"
                    )
                _sa = _stack.block[0].layer[0].SelfAttention
                if (
                    _sa.qkv_weight.shape != (12288, 4096)
                    or _sa.o.weight.shape != (4096, 4096)
                    or tuple(_sa._pos_bias.shape[-2:]) != (128, 128)
                ):
                    raise NotImplementedError(
                        "--cu-fused-text-encoder attention kernel has fixed 64 heads x 64 dims at compile time (4096->12288->4096), "
                        f"this checkpoint is {tuple(_sa.qkv_weight.shape)} / "
                        f"{tuple(_sa.o.weight.shape)}"
                    )
                trunner = FusedTextRunner(_stack, torch.device(self.device), split=text_fused_split)
                trunner.install(_stack)
                arunner = FusedAttnRunner(_stack, torch.device(self.device), split=text_fused_split)
                arunner.install(_stack)
                self._text_fused_runner = trunner
                self._text_fused_attn_runner = arunner
                # plain attr (not a Module/param): not in state_dict, unaffected by .to/.eval
                self.text_encoder._text_fused_runner = trunner
                self.text_encoder._text_fused_attn_runner = arunner
                log.info(
                    "cu-fused-text-encoder: UMT5 attention + FFN fused kernels "
                    + (
                        "non-cooperative split form (5+4 launches/layer, no full-GPU co-residency required)"
                        if text_fused_split
                        else "cooperative kernel form (2 launches/layer, full-GPU co-residency required)"
                    )
                    + f" ready ({time.perf_counter() - _t0:.1f}s); only the S_PHASE_1/F_PHASE_1 row reduction order differs from the fp8 "
                    f"reference (measured ~0.0005% activation diff of 1 grid cell)"
                )
            self._text_encoder_device = self.device
            emb_loc = "CPU" if text_emb_cpu else "GPU"
            if text_resident:
                log.info(f"text encoder fp8-resident ({n_fp8} Linear, direct load) on GPU, embedding on {emb_loc}")
            else:
                reason = "--text-bf16" if not text_fp8 else ("no fp8 file" if not te_has_fp8 else "cpu-only")
                log.info(f"text encoder bf16 on GPU (fp8 off: {reason}), embedding on {emb_loc}")
        else:
            self.text_encoder.to("cpu")
            self._text_encoder_device = "cpu"
        log.info(f"text encoder on {self._text_encoder_device} ({time.perf_counter() - t0:.1f}s)")
        if not te_on_gpu and self.device.startswith("cuda"):
            log.info(
                "note: text encoder runs on CPU (bf16 weights ~13 GB main RAM); "
                "pass `--text-encoder-device cuda` to load it fp8-resident on GPU"
            )

        t0 = time.perf_counter()
        log.info(f"loading Wan VAE encoder from {self.config.vae_dir} ...")
        self.vae = load_wan_vae_encoder(self.config.vae_dir, dtype=torch_dtype)
        self.vae.to(self.device)
        log.info(f"vae encoder on {self.device} ({time.perf_counter() - t0:.1f}s)")

        self.tokenizer = UMT5Tokenizer(self.config.tokenizer_dir, max_length=self.config.tokenizer_max_len)

        # ------------------------------------------------------------------ #
        # policy module + checkpoint weights
        # ------------------------------------------------------------------ #
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            core = FastWAM(
                self.config,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch_dtype,
                action_context_cache=action_context_cache,
            )
        finally:
            torch.set_default_dtype(default_dtype)
        self.policy = FastWAMPolicy(core, self.config)
        # hand the overlap flag to the DiT core, which decides whether to fuse step 0
        # into the video prefill; see ``FastWAM._prefill_video_layer_action_layer_overlap``.
        self.policy.model.overlap = self.overlap
        # ``--no-prompt-cache``: force a real UMT5 forward per request; set before warmup so
        # the warmup call does not seed the memo either (see ``FastWAM.encode_prompt``).
        self.cache_prompt = bool(cache_prompt)
        self.policy.model.cache_prompt = self.cache_prompt
        # Where the UMT5 forward actually runs (its embedding table may be on CPU).
        self.policy.model.text_encoder_device = self._text_encoder_device

        t0 = time.perf_counter()
        log.info(f"loading fastwam weights from {checkpoint_dir} ...")
        # fp8-resident direct load: swap in FP8Linear shells first, then the loader copies
        # fp8 weights + per-row scale straight from the checkpoint (no dequant/requant).
        fp8_files = fp8_checkpoint_files(checkpoint_dir)
        has_fp8 = bool(fp8_files)
        if has_fp8:
            log.info(f"fp8 checkpoint detected: {[os.path.basename(f) for f in fp8_files]}")
        elif video_fp8 or action_fp8:
            log.info("note: fp8 requested but no fp8 files in dir -> experts stay bf16")
        video_fp8 = video_fp8 and has_fp8
        action_fp8 = action_fp8 and has_fp8
        if video_fp8:
            n_layers = fp8ify_structural(self.policy.model.mot.mixtures["video"], min_dim=256)
            log.info(
                f"video-fp8: {n_layers} Linear layers of the video expert fp8-resident (direct load, ~5GB VRAM saved)"
            )
        if action_fp8:
            n_layers = fp8ify_structural(self.policy.model.mot.mixtures["action"], min_dim=256)
            log.info(
                f"action-fp8: {n_layers} Linear layers of the action expert fp8-resident (direct load, ~1GB VRAM saved)"
            )
        skipped = load_model(self.policy.model, checkpoint_dir)
        if skipped:
            raise RuntimeError(
                f"{len(skipped)} unmatched weights (e.g. {skipped[0]}); check that the "
                "checkpoint layout matches the fastwam backend module tree"
            )
        log.info(f"weights loaded in {time.perf_counter() - t0:.1f}s")

        if pack_qkv:
            # Packing follows each expert's fp8/bf16 tier and must run after loading.
            video_packed = (
                pack_attention_qkv_fp8(self.policy.model.mot.mixtures["video"])
                if video_fp8
                else pack_attention_qkv(self.policy.model.mot.mixtures["video"])
            )
            action_packed = (
                pack_attention_qkv_fp8(self.policy.model.mot.mixtures["action"])
                if action_fp8
                else pack_attention_qkv(self.policy.model.mot.mixtures["action"])
            )
            log.info(
                f"pack-qkv: video({'fp8' if video_fp8 else 'bf16'}) {video_packed} + "
                f"action({'fp8' if action_fp8 else 'bf16'}) {action_packed} projections packed"
            )

        # --cu-fused-adit: three fused CUDA kernels (self-attn / cross-attn / FFN)
        # replace the action denoise layer. Needs action fp8-resident, strided context
        # k/v, CUDA and sm_89+. If unmet, degrade (fused -> split -> eager) unless
        # --require-fused.
        if action_fused:
            blockers = []
            if not self.device.startswith("cuda"):
                blockers.append("CUDA device required (fused kernels are hand-written CUDA)")
            else:
                cap = tuple(torch.cuda.get_device_capability(self.device))
                if cap < (8, 9):
                    blockers.append(f"requires sm_89+ (fp8 e4m3 mma), this machine is sm_{cap[0]}{cap[1]}")
            if not action_fp8:
                blockers.append("action expert is not fp8-resident (--action-bf16 cannot coexist with the fused tier)")
            if not action_context_cache:
                blockers.append(
                    "requires action cross-attention k/v cache "
                    "(--no-action-context-cache cannot coexist with the fused tier)"
                )
            if self._fused_or_eager("action fusion", blockers, require_fused):
                action_fused_split = self._resolve_split("action", action_fused_split, require_fused)
                # WARNING: packing is destructive (frees the original q/k/v), so every
                # degradation decision must be made before it -- after this eager is gone.
                if not pack_qkv:
                    n = pack_attention_qkv_fp8(self.policy.model.mot.mixtures["action"])
                    log.info(f"action-fused: auto pack-qkv (fp8) {n} action projections")
                log.info(
                    "action-fused: bf16-flash tier (self-attn / cross-attn / FFN fused kernels; chunk drift ~2% "
                    "rel-RMS)"
                )
            else:
                action_fused = False

        # --cu-fused-vdit: fused kernels replace the video prefill block (attn self+cross
        # + FFN tail); both forms are bitwise-identical. ``video_fused_split=True``
        # (default) is the non-cooperative form (no full-GPU co-residency required);
        # ``False`` (--cooperative-kernel) needs the whole grid to co-reside.
        if video_fused:
            _vid_blocks = self.policy.model.mot.mixtures["video"].blocks
            blockers = []
            if not self.device.startswith("cuda"):
                blockers.append("CUDA device required (fused kernels are hand-written CUDA)")
            else:
                cap = tuple(torch.cuda.get_device_capability(self.device))
                if cap < (8, 9):
                    blockers.append(f"requires sm_89+ (fp8 e4m3 mma), this machine is sm_{cap[0]}{cap[1]}")
            if not video_fp8:
                blockers.append("video expert is not fp8-resident (--video-bf16 cannot coexist with the fused tier)")
            # Kernel constants are fixed at compile time (a mismatch is an explicit error);
            # the token count is only known at inference time, so FusedVideoRunner checks it.
            hidden = _vid_blocks[0].ffn[0].in_features
            ffn_dim = _vid_blocks[0].ffn[0].out_features
            if (hidden, ffn_dim) != (3072, 14336):
                blockers.append(
                    f"kernels have fixed hidden=3072/ffn=14336 at compile time, this checkpoint is {hidden}/{ffn_dim}"
                )
            block0 = _vid_blocks[0]
            if (block0.num_heads, block0.attn_head_dim) != (24, 128):
                blockers.append(
                    f"attention kernel has fixed 24 heads x 128 (=3072 wide) at compile time, this checkpoint is "
                    f"{block0.num_heads}x{block0.attn_head_dim}"
                )
            _ctx_len = getattr(self.config, "context_len", None)
            if _ctx_len is not None and int(_ctx_len) + 1 != 129:
                blockers.append(
                    f"cross kernel has fixed context=129 at compile time ({_ctx_len} text + 1 proprio), "
                    f"this checkpoint is {int(_ctx_len) + 1}"
                )
            if self._fused_or_eager("video fusion", blockers, require_fused):
                video_fused_split = self._resolve_split("video", video_fused_split, require_fused)
                log.info(
                    "cu-fused-vdit: attention(self+cross) + FFN fused kernels "
                    + (
                        "non-cooperative split form (16 launches/layer, no full-GPU co-residency required)"
                        if video_fused_split
                        else "cooperative kernel form (3 launches/layer, full-GPU co-residency required)"
                    )
                    + "; quantization-drift tier, rel <=3e-2"
                )
            else:
                video_fused = False

        self.policy.to(self.device).eval()

        if action_fused:
            from .models.fused_action import FusedActionRunner

            t0 = time.perf_counter()
            runner = FusedActionRunner(
                self.policy.model.mot.mixtures["action"].blocks,
                torch.device(self.device),
                split=action_fused_split,
            )
            self._action_fused_runner = runner
            # plain attr (not a Module/param): not in state_dict, unaffected by .to/.eval
            self.policy.model._action_fused_runner = runner
            log.info(
                "cu-fused-adit: self-attn / cross-attn / FFN fused kernels "
                + (
                    "non-cooperative split form (6+6+4 launches/layer, no full-GPU co-residency required)"
                    if action_fused_split
                    else "cooperative kernel form (3 launches/layer, full-GPU co-residency required)"
                )
                + f" ready ({time.perf_counter() - t0:.1f}s)"
            )

        if action_pre_fused:
            # Fused action time path + context kv precompute; context is bitwise-identical,
            # the time path differs only in fp32 reduction order (sub-ulp in bf16).
            # Degrades to skipping this optimization when it cannot run.
            _pre_blockers = []
            if not self.device.startswith("cuda"):
                _pre_blockers.append("CUDA device required (fused kernels are hand-written CUDA)")
            else:
                cap = tuple(torch.cuda.get_device_capability(self.device))
                if cap < (8, 9):
                    _pre_blockers.append(f"requires sm_89+, this machine is sm_{cap[0]}{cap[1]}")
            if not action_fused:
                _pre_blockers.append(
                    "requires --cu-fused-adit (provides fp8-resident + the auto "
                    "pack-qkv cross_attn.kv; --pack-qkv also works)"
                )
            if self._fused_or_eager("action-pre fusion", _pre_blockers, require_fused):
                from .models.fused_action_pre import ActionPreRunner

                t0 = time.perf_counter()
                action_expert = self.policy.model.mot.mixtures["action"]
                action_pre_runner = ActionPreRunner(action_expert, torch.device(self.device))
                action_pre_runner.install(action_expert)
                self._action_pre_runner = action_pre_runner
                self.policy.model._action_pre_runner = action_pre_runner
                log.info(
                    f"action-pre-fused: time path + context kv fused kernels ready ({time.perf_counter() - t0:.1f}s)"
                )

        if video_pre_fused:
            # Purely exact (bitwise-identical): resident freqs/masks + a prebuilt fp64
            # sinusoidal frequency table.
            from .models.fused_video_pre import VideoPreRunner

            t0 = time.perf_counter()
            pre_v = VideoPreRunner(self.policy.model)
            pre_v.install()
            self._video_pre_runner = pre_v
            log.info(f"video-pre-fused: video pre-cache + exact time embedding ready ({time.perf_counter() - t0:.1f}s)")

        if video_fused:
            from .models.fused_video import FusedAttnRunner, FusedVideoRunner

            t0 = time.perf_counter()
            blocks = self.policy.model.mot.mixtures["video"].blocks
            # The attention section and the FFN tail install independent hooks
            # (``_fused_attn``: self+cross; ``_fused_ffn_tail``: norm2+modulate+FFN+gate).
            arunner = FusedAttnRunner(blocks, torch.device(self.device), split=video_fused_split)
            arunner.install(blocks)
            self._video_fused_attn_runner = arunner
            # The model also holds the runner: CUDA graph capture needs it to suppress D2H
            # synchronization (a host sync during capture is illegal).
            self.policy.model._video_fused_attn_runner = arunner
            vrunner = FusedVideoRunner(blocks, torch.device(self.device), split=video_fused_split)
            vrunner.install(blocks)
            self._video_fused_runner = vrunner
            # plain attr (not a Module/param): not in state_dict, unaffected by .to/.eval
            self.policy.model._video_fused_runner = vrunner
            log.info(f"cu-fused-vdit: attn(self+cross) + FFN fused kernels ready ({time.perf_counter() - t0:.1f}s)")

        # ------------------------------------------------------------------ #
        # pre/post processors (MIN_MAX normalization)
        # ------------------------------------------------------------------ #
        self.normalizer = load_normalizer(checkpoint_dir, self.config.normalization_mapping)
        self.unnormalizer = load_unnormalizer(checkpoint_dir, self.config.normalization_mapping)
        self.preprocessor = PreProcessor(self.config, self.normalizer, device=self.device, camera_alias=camera_alias)
        self.postprocessor = PostProcessor(self.config, self.unnormalizer, device="cpu")

        if warmup:
            self._warmup()
        if self.compile_model:
            self._install_compile()
        if self.graph_enabled:
            self._setup_graph()
        # --profile: one-shot startup latency report through the configured path
        # (the graph, when enabled), printed for worker/serve logs.
        if profile:
            try:
                self.profile()
            except Exception as e:  # noqa: BLE001 - never fail startup on a diagnostic
                log.warning(f"profile failed: {e}")

    # ------------------------------------------------------------------ #
    def _dummy_frame(self):
        """Zero frame for warmup / graph capture (image keys already carry the prefix)."""
        spec = self.describe()
        h, w = spec["resize"][1], spec["resize"][0]
        # ``base.make_frame`` adds the "observation.images." prefix itself, but the
        # checkpoint's keys usually already carry it -> strip it to avoid a double prefix.
        images = {
            cam.split("observation.images.")[-1]: torch.zeros(3, h, w, dtype=torch.float32)
            for cam in self.config.image_feature_keys
        }
        state = torch.zeros(self.config.proprio_dim or 8, dtype=torch.float32)
        return self.make_frame(images, state, "warmup")

    def _install_compile(self) -> None:
        """Apply ``torch.compile`` to the DiT core's two hot regions.

        Regions rather than ``FastWAM._denoise_core`` as a whole: the core's step loop is a
        Python ``for``, so compiling the whole core would re-trace the step body once per
        step. Installed *after* the eager warmup so the defensive ``validate`` syncs and the
        kernel JIT run once on the real eager path, and the compiled regions then see exactly
        the arguments production passes.
        """
        model = self.policy.model
        t0 = time.perf_counter()
        # ``dynamic=False``: the core's shapes are fixed by the checkpoint/config, so static
        # kernels are fastest and no recompile can happen in flight. ``compile_region``
        # returns the region's attribute path, which is what we log.
        self.compiled_regions = [
            compile_region(model, model.mot, "prefill_video_cache", dynamic=False),
            compile_region(model, model, "denoise_step", dynamic=False),
        ]
        log.info(
            f"--compile: torch.compile on the DiT core regions "
            f"({', '.join(self.compiled_regions)}; the denoise loop "
            f"stays eager so the step body is compiled once instead of once per step)"
        )
        # With ``--graph`` the capture drives the regions first anyway (with the no-sync
        # arguments), so compiling now would only add a second, unused specialisation.
        if not self.graph_enabled:
            self.predict_action_chunk(self._dummy_frame())  # compile now, not on request #1
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        log.info(
            f"--compile: DiT core regions compiled in {time.perf_counter() - t0:.1f}s; "
            f"the eager path is unchanged (compiled numerics may drift)"
        )

    def _setup_graph(self) -> None:
        """Attach the CUDA-graph runner and capture the default key with one dummy request."""
        from .models.graph_runner import FastWAMGraphRunner

        t0 = time.perf_counter()
        # ONE graph serves every camera count: the image slots are concatenated into a single
        # frame before the captured core.
        log.info(
            "capturing fastwam CUDA graph: one multi-stream graph for the DiT core "
            "(every camera count shares it; the image slots are concatenated into one "
            "frame before the graph)"
        )
        self._graph_runner = FastWAMGraphRunner(self.policy.model, self.device, overlap=self.overlap)
        self.policy.model._graph_runner = self._graph_runner
        self.predict_action_chunk(self._dummy_frame())  # triggers the capture
        torch.cuda.synchronize()
        log.info(
            f"fastwam CUDA graph ready ({self._graph_runner.num_graphs} graph(s), {time.perf_counter() - t0:.2f}s)"
        )

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        """Run one dummy inference so CUDA init / lazy loads don't inflate first-request latency."""
        t0 = time.perf_counter()
        self.predict_action_chunk(self._dummy_frame())
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        log.info(f"warmup done in {time.perf_counter() - t0:.1f}s")

    # fused-tier degradation
    #
    # ``--cu-fused-*`` is a **preference**, not a requirement: if it cannot run, fall back
    # to the next tier (fused cooperative -> fused split -> eager) and log the reason;
    # ``--require-fused`` restores hard failure.
    #
    # WARNING: degrade only **before destructive packing**. ``pack_fused`` / pack-qkv frees
    #    the original q/k/v and wi_0/wi_1 weights, after which the eager path is already
    #    wrong. All degradation decisions precede packing; later failures are raised as usual.
    # ------------------------------------------------------------------ #
    _FAMILY_KERNELS = {
        "text-encoder": ("tmt5.ffn", "tmt5.attn"),
        "action": ("adit.attn_self", "adit.attn_cross", "adit.ffn"),
        "video": ("vdit.attn_self", "vdit.attn_cross", "vdit.ffn"),
    }

    def _fused_or_eager(self, label: str, blockers: list[str], require: bool) -> bool:
        """Return whether the fused tier can be used, degrading to eager otherwise.

        An empty ``blockers`` means it can; ``require`` raises instead of degrading.
        """
        if not blockers:
            return True
        why = "; ".join(blockers)
        if require:
            raise NotImplementedError(f"{label}: fused tier unavailable -- {why}")
        log.warning(
            f"{label}: fused tier unavailable -> falling back to eager ({why})"
            f"; add --require-fused to make it a hard failure"
        )
        return False

    def _resolve_split(self, family: str, want_split: bool, require: bool) -> bool:
        """Return whether to use the non-cooperative split form, switching when needed."""
        if want_split:
            return True
        from .kernels import registry as reg

        for name in self._FAMILY_KERNELS[family]:
            ok, why = reg.coop_available(reg.spec(name), self.device)
            if ok:
                continue
            if require:
                raise NotImplementedError(f"{family} cooperative form unavailable -- {name}: {why}")
            log.warning(
                f"{family}: cooperative form unavailable -> switching to non-cooperative split"
                f" ({name}: {why}); add --require-fused to make it a hard failure"
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # interface
    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        height, width = self.config.image_size
        return {
            "model_type": "fastwam",
            "cameras": self.config.image_feature_keys,
            "resize": [width, height],  # (w, h) for the gateway
            "action_dim": self.config.action_dim,
            "chunk_size": self.config.action_horizon,
            "pad_mode": "stretch",
        }

    def _batch_from_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        return self.preprocessor(frame)

    def _prepare_noise(self, noise: torch.Tensor | None) -> torch.Tensor | None:
        if noise is None:
            return None
        noise = noise.detach().to(device="cpu", dtype=torch.float32)
        return noise

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_action_chunk(
        self,
        frame: dict[str, Any],
        noise: torch.Tensor | None = None,
        return_normalized: bool = False,
    ) -> Any:
        """Return the full unnormalized action chunk ``[chunk_size, action_dim]`` (numpy).

        ``return_normalized=True`` also returns the normalized chunk, as a tuple.
        """
        batch = self._batch_from_frame(frame)
        noise = self._prepare_noise(noise)
        actions = self.policy.predict_action_chunk(
            input_image=batch["input_image"],
            state=batch.get("state"),
            task=batch["task"],
            noise=noise,
            num_inference_steps=self.config.num_inference_steps,
        )  # [horizon, action_dim] fp32 cpu normalized
        chunk = self.postprocessor(actions)
        if return_normalized:
            return chunk.cpu().numpy(), actions.detach().cpu().numpy()
        return chunk.cpu().numpy()

    @torch.inference_mode()
    def select_action(self, frame: dict[str, Any], noise: torch.Tensor | None = None) -> np.ndarray:
        """Return the single next action ``(action_dim,)``, unnormalized (numpy)."""
        batch = self._batch_from_frame(frame)
        noise = self._prepare_noise(noise)
        action = self.policy.select_action(
            input_image=batch["input_image"],
            state=batch.get("state"),
            task=batch["task"],
            noise=noise,
            num_inference_steps=self.config.num_inference_steps,
        )
        action = self.postprocessor(action)
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self, reference_dir: str) -> dict[str, float]:
        """Compare the engine outputs against reference dumps.

        Raises:
            NotImplementedError: fastwam reference dumps are not available yet.
        """
        raise NotImplementedError(
            "fastwam reference dumps are not available yet; the full-chain harness writes into "
            f"{reference_dir!r} and needs a GPU with ~26 GB+ free VRAM."
        )


def print_validation_report(errors: dict[str, float]) -> None:
    """Print the per-component validation error report (fastwam)."""
    print("fastwam validation errors:")
    for name, err in sorted(errors.items()):
        print(f"  {name}: {err:.6e}")
