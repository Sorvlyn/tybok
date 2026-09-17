"""pi0.5 policy engine (registered as ``model_type="pi05"``).

Loads the checkpoint and exposes the inference API: ``select_action`` (one
queued action per call) and ``predict_action_chunk`` (the whole chunk). The
model is plain PyTorch ``nn.Module`` code.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ...base import PolicyEngine, compile_region
from ...graph import (
    CaptureSession,
    GraphRunner,
    camera_counts,
)
from ...registry import register, reject_unsupported_flags
from ..rtc import RTCConfig, RTCProcessor
from .config import PI05Config
from .models.flow_matching import (  # noqa: E402  (kept grouped with the other model imports)
    create_sinusoidal_pos_embedding,
)
from .models.policy import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    PI05Policy,
    load_model,
)
from .preprocess import PostProcessor, PreProcessor, load_normalizer, load_unnormalizer
from .tokenizer import PaliGemmaTokenizer

log = logging.getLogger("tybok.pi05")

# The three model components the captured graph bakes in, as ``<role>=<module path
# in the modeling files>``; one graph per shape covers all three.
_GRAPH_COMPONENTS = (
    "vision encoder=paligemma.model.vision_tower.vision_model, "
    "llm backbone=paligemma.model.language_model, "
    "action expert=gemma_expert.model"
)


def _require_fp8_mma(device: str, flag_names: list[str]) -> None:
    """Fail loudly when an fp8 flag is set on hardware without fp8 tensor cores.

    The fp8 (W8A8) Triton kernels need the e4m3 tensor-core MMA, which only
    exists on sm_89+ (Ada / Hopper / Blackwell); raising here beats a low-level
    Triton compile/launch failure during warmup.
    """
    if not flag_names:
        return
    flags = " / ".join(flag_names)
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            f"{flags} require fp8 (e4m3) tensor-core MMA and a CUDA device, but the engine is running on {device!r}"
        )
    cap = torch.cuda.get_device_capability(device)
    if cap < (8, 9):
        raise RuntimeError(
            f"{flags} require sm_89+ (fp8 e4m3 tensor-core MMA), but this device "
            f"is sm_{cap[0]}{cap[1]}. Drop the fp8 flag(s), or run on sm_89+ "
            f"(Ada / Hopper / Blackwell)."
        )


@register("pi05")
class PI05Engine(PolicyEngine):
    # Flags this backend does not implement (owned by smolvla / fastwam): the inert
    # defaults the worker forwards; a different value is rejected by the registry.
    _UNSUPPORTED_FLAGS = {
        # smolvla-only
        "tl_llm_fused_attn": False,
        "tl_vit_oproj": False,
        # fastwam-only
        "pack_qkv": False,
        "cache_prompt": True,
        "action_context_cache": True,
        "text_fused": False,
        "text_fused_split": False,
        "video_fp8": True,
        "action_fp8": True,
        "text_fp8": True,
        "text_emb_cpu": True,
        "action_fused": False,
        "action_fused_split": False,
        "video_fused": False,
        "video_fused_split": False,
        "action_pre_fused": False,
        "video_pre_fused": False,
        "require_fused": False,
        "text_encoder_dir": None,
        "vae_dir": None,
        "text_encoder_device": "cpu",
    }

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
        # Accepted for CLI uniformity, intentionally unused: pi0.5's prefix-KV reuse is
        # structural, and upstream lerobot's pi05 has no equivalent knob either.
        cache_expert_prefix_kv: bool = True,
        profile: bool = False,
        overlap: bool = True,
        tl_fused_vit: bool = False,
        tl_llm_flash_attn: bool = False,
        tl_fused_expert: bool = False,
        tl_fp8_llm_mlp: bool = False,
        tl_fp8_expert_mlp: bool = False,
        vit_mlp_dtype: str | None = None,
        skip_empty_images: bool = False,
        pad_free: bool = False,
        camera_alias: dict[str, str] | None = None,
        tokenizer_dir: str | None = None,
        rtc_config: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(checkpoint_dir, device=device)
        # ``create_engine`` already rejects these on the raw kwargs; re-check here
        # so direct construction (bypassing the registry) fails the same way.
        reject_unsupported_flags(type(self), "pi05", kwargs)
        if sampler not in ("euler",):
            raise ValueError(
                f"sampler {sampler!r} not supported for pi05 yet (only 'euler' is bit-exact against the reference)"
            )
        if not cache_expert_prefix_kv:
            # Accepted but meaningless here (see the parameter's comment): say so out
            # loud instead of silently discarding a user request.
            log.info(
                "--no-expert-prefix-kv-cache: no-op for pi05 (the prefix K/V reuse is structural -- "
                "prefill builds the prefix K/V cache and every denoising step only reads it)"
            )
        # ``--compile`` is eager-only for pi05 (applied after the weight load, below).
        # The fp8 tiers need the sm_89+ e4m3 MMA: check before loading weights so an
        # unsupported GPU fails fast instead of inside Triton at warmup.
        _require_fp8_mma(
            self.device,
            [
                name
                for name, on in (
                    ("--tl-fp8-llm-mlp", tl_fp8_llm_mlp),
                    ("--tl-fp8-expert-mlp", tl_fp8_expert_mlp),
                )
                if on
            ],
        )
        self.config = PI05Config.from_pretrained(checkpoint_dir, tokenizer_dir=tokenizer_dir)
        if num_steps is not None:
            self.config.num_steps = int(num_steps)
        self.tokenizer = PaliGemmaTokenizer(self.config.tokenizer_dir, max_length=self.config.tokenizer_max_length)

        # Real-Time Chunking (RTC): pure guidance math, no extra weights; off by
        # default, config from the checkpoint ``rtc_config`` block and/or CLI overrides.
        self.rtc_processor = None
        rtc_cfg = self._build_rtc_config(rtc_config)
        if rtc_cfg is not None:
            if graph:
                raise ValueError(
                    "--rtc and --graph are mutually exclusive: RTC guidance runs its "
                    "denoising under torch.enable_grad(), and any graph-external "
                    "autograd execution corrupts the captured CUDA-graph memory "
                    "pool (the graph would silently return wrong chunks afterwards)"
                )
            self.rtc_processor = RTCProcessor(rtc_cfg)
            log.info(f"RTC enabled: {rtc_cfg.to_dict()}")

        # Build under a bf16 default dtype (VLM/expert layers); the vision tower,
        # norms, AdaRMS modulations and action/time projections stay fp32 explicitly.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.policy = PI05Policy(self.config, rtc_processor=self.rtc_processor)
        finally:
            torch.set_default_dtype(default_dtype)

        skipped = load_model(self.policy, checkpoint_dir)
        if skipped:
            log.info(f"skipped {len(skipped)} unmatched weights (e.g. {skipped[0]})")

        self.policy.to(self.device).eval()

        # --tl-fused-vit (drift tier): fused SigLIP tower attention + bf16 MLP +
        # post_layernorm+projector. Applied after weight load, before warmup / capture.
        self.tl_fused_vit = bool(tl_fused_vit)
        if self.tl_fused_vit:
            from .models.paligemma_with_expert import PaliGemmaModel
            from .models.vision import SiglipVisionTransformer

            vlm: PaliGemmaModel = self.policy.model.paligemma_with_expert.paligemma.model
            vision = cast(SiglipVisionTransformer, vlm.vision_tower.vision_model)
            vision.set_fused_attn()
            vision.set_fused_mlp()
            vlm.set_fused_projector()
            log.info("vision tower: fused Triton attention + MLP + projector enabled (drift tier)")

        # --vit-mlp-dtype {fp16,bf16} (drift tier): only the SigLIP tower MLP GEMMs
        # (fc1/fc2) change dtype; norms, attention and residuals stay fp32. Inert under
        # --tl-fused-vit. Applied after weight load, before warmup / capture.
        self.vit_mlp_dtype: torch.dtype | None = None
        if vit_mlp_dtype is not None:
            if vit_mlp_dtype not in ("fp16", "bf16"):
                raise ValueError(f"--vit-mlp-dtype must be 'fp16' or 'bf16', got {vit_mlp_dtype!r}")
            if self.tl_fused_vit:
                log.info("--vit-mlp-dtype ignored: --tl-fused-vit owns the fused ViT MLP precision")
            else:
                dt = torch.float16 if vit_mlp_dtype == "fp16" else torch.bfloat16
                from .models.paligemma_with_expert import PaliGemmaModel
                from .models.vision import SiglipVisionTransformer

                vlm = cast(PaliGemmaModel, self.policy.model.paligemma_with_expert.paligemma.model)
                vision = cast(SiglipVisionTransformer, vlm.vision_tower.vision_model)
                vision.set_mlp_dtype(dt)
                self.vit_mlp_dtype = dt
                log.info(f"vision tower: MLP-only {vit_mlp_dtype} enabled (drift tier)")

        # --skip-empty-cams: skip the SigLIP ViT forward for empty camera slots
        # (mask=0, all-(-1) pixels); their tokens are masked out of every attention, so
        # a zeros placeholder is bit-identical. Inert under ``--graph``, which bakes
        # all-True masks at capture; set before warmup / capture.
        self.skip_empty_images = bool(skip_empty_images)
        if self.skip_empty_images:
            self.policy.model.skip_empty_images = True

        # --pad-free (drift tier): padding-free VLM prefill.
        #   - eager: compact the prefix to live tokens (drop empty-camera 256-blocks AND
        #     language pads) -- WARNING: ``live_idx`` is a GPU->host sync, illegal in a graph.
        #   - graph: skip only the always-empty ``empty_camera_*`` slots (fixed prefix
        #     length); less aggressive, as language pads and empty real cams remain.
        self.pad_free = bool(pad_free)
        if self.pad_free:
            self.policy.model.pad_free = True
            if graph:
                log.info(
                    "padding-free prefill (graph): skipping empty camera slots per "
                    f"camera count -> prefix {256 * len(self.config.real_camera_keys) + self.config.tokenizer_max_length} "
                    f"at full count (was 968) [drift tier]"
                )
            else:
                log.info("padding-free prefill enabled (drift tier, eager: live-only compaction)")

        # --tl-fused-expert (drift tier): expert denoising attention + MLP as fused
        # Triton kernels; --tl-fp8-expert-mlp swaps the MLP for the W8A8 fp8 chain.
        # Orthogonal to --graph (the kernels are captured), not combinable with --compile.
        # Set before warmup / capture so the kernels are JIT-compiled and baked.
        self.tl_fused_expert = bool(tl_fused_expert)
        self.tl_fp8_expert_mlp = bool(tl_fp8_expert_mlp)
        if self.tl_fused_expert or self.tl_fp8_expert_mlp:
            from .models.rope import build_rope_tables

            head_dim = self.config.expert.head_dim  # 256
            cos, sin = build_rope_tables(head_dim, 2048, self.device)
            ctx: dict[str, Any] = {"cos": cos, "sin": sin, "eps": 1e-6}
            model = self.policy.model
            if self.tl_fused_expert:
                # Fuse the expert final norm + action_out_proj, letting the decoder skip
                # its final norm. Tied to the attention/MLP fusion, not to the fp8-only
                # MLP flag, which must not change this tail.
                model.fuse_final_tail = True
                model.paligemma_with_expert.gemma_expert.model.skip_final_norm = True
            for layer in model.paligemma_with_expert.gemma_expert.model.layers:
                layer.fused_attn = self.tl_fused_expert
                layer.fused_mlp = self.tl_fused_expert and not self.tl_fp8_expert_mlp
                layer.use_fp8_mlp = self.tl_fp8_expert_mlp
                layer.triton_ctx = ctx
            if self.tl_fused_expert:
                log.info(
                    "expert attention/mlp: fused Triton denoise enabled "
                    "(AdaRMS norm + interleaved qkv GEMM+RoPE + split-K flash; "
                    "norm+gate/up+GELU; drift tier)"
                )
            if self.tl_fp8_expert_mlp:
                log.info("expert MLP: W8A8 fp8 chain enabled (fused fp8 gate/up + fp8 down; drift tier)")

        # --tl-fp8-llm-mlp (drift tier): VLM prefill MLP (gate/up/down) as the fused
        # Triton W8A8 fp8 GEMM. Applied before warmup / capture so the kernels are
        # baked; orthogonal to the other --tl-fused-* / --pad-free tiers.
        self.tl_fp8_llm_mlp = bool(tl_fp8_llm_mlp)
        if self.tl_fp8_llm_mlp:
            from .models.triton_prefill_mlp import make_fp8_mlp_forward

            vlm_layers = self.policy.model.paligemma_with_expert.paligemma.model.language_model.layers
            for layer in vlm_layers:
                layer.mlp.forward = make_fp8_mlp_forward(layer.mlp)  # type: ignore[method-assign]
            log.info("VLM prefill MLP: fused Triton fp8 enabled (drift tier)")

        # --tl-llm-flash-attn (drift tier): q/k/v in one cuBLAS GEMM + Triton GQA
        # flash prefill attention; input_layernorm, RoPE and o_proj stay eager.
        self.tl_llm_flash_attn = bool(tl_llm_flash_attn)
        if self.tl_llm_flash_attn:
            vlm_layers = self.policy.model.paligemma_with_expert.paligemma.model.language_model.layers
            for layer in vlm_layers:
                layer.self_attn.fused_prefill_attn = True
            log.info("VLM prefill attention: concat qkv GEMM (cuBLAS) + Triton GQA flash enabled (drift tier)")

        norm_map = self.config.normalization_mapping
        # --camera-alias: rename robot/dataset cameras to the checkpoint's image slots
        # before the batch is built, e.g. ``{"wrist_image": "image2"}`` (full
        # ``observation.images.*`` keys accepted). Only when the target slot is missing.
        self.camera_alias: dict[str, str] | None = None
        if camera_alias:
            img_feats = [k for k in self.config.image_features if "images" in k]
            full: dict[str, str] = {}
            for src, dst in camera_alias.items():
                src_key = src if src.startswith("observation.images.") else f"observation.images.{src}"
                dst_key = dst if dst.startswith("observation.images.") else f"observation.images.{dst}"
                if dst_key not in img_feats:
                    raise ValueError(
                        f"camera_alias target {dst!r} is not an expected image feature "
                        f"(available: {[k[18:] for k in img_feats]})"
                    )
                if src_key in img_feats:
                    raise ValueError(
                        f"camera_alias source {src!r} is already an expected image feature "
                        f"({[k[18:] for k in img_feats]})"
                    )
                if src_key in full and full[src_key] != dst_key:
                    raise ValueError(f"camera_alias maps {src!r} twice to different targets")
                full[src_key] = dst_key
            self.camera_alias = full
            log.info(f"camera alias: {dict(camera_alias)}")
        self.preprocessor = PreProcessor(
            self.config,
            self.tokenizer,
            load_normalizer(checkpoint_dir, norm_map),
            device=self.device,
            camera_alias=self.camera_alias,
        )
        self.postprocessor = PostProcessor(self.config, load_unnormalizer(checkpoint_dir, norm_map), device="cpu")

        # select_action FIFO queue
        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.config.n_action_steps)

        self.seed = seed
        self._noise_gen: torch.Generator | None = None
        if seed is not None:
            self._noise_gen = torch.Generator(device=self.device).manual_seed(int(seed))
            self.policy.model.noise_generator = self._noise_gen

        # ``--graph`` is implemented for pi05 (single CUDA graph per shape: the
        # prefix length is a compile-time constant, 256*len(image_features) + 200).
        self.graph_requested = bool(graph)
        self.graph_enabled = self.graph_requested and str(self.device).startswith("cuda")

        # --skip-empty-cams reports the mode it actually runs in: the check is a D2H
        # ``img_mask.all()`` sync, illegal inside a capture and never reached under
        # ``use_static`` (captured masks are all-True), so it is inert on the graph path.
        if self.skip_empty_images:
            if self.graph_enabled:
                log.info(
                    "--skip-empty-cams is inert on the --graph path (the capture bakes "
                    "all-True masks, so the empty-camera check never runs) -- use "
                    "--pad-free to drop the empty slots from the graph prefix instead"
                )
            else:
                log.info("empty-camera slots skip the ViT forward (bit-exact)")

        # ``--compile`` compiles the inference *phases*, not ``sample_actions``: the denoise
        # loop body runs once per step, so one region would hold one copy per step for little
        # gain. The sequential ``--graph`` capture calls ``sample_actions``, so it captures
        # these regions, while the overlap fusion captures per-layer graphs that never call
        # them -- overlap is therefore disabled below, not refused. Mutually exclusive with
        # the fused Triton tiers: inductor precompiles those kernels in configurations the
        # eager launcher never uses, failing with a deep ``CompilationError``.
        fused_flags = [
            name
            for name, on in (
                ("--tl-fused-vit", tl_fused_vit),
                ("--tl-llm-flash-attn", tl_llm_flash_attn),
                ("--tl-fused-expert", tl_fused_expert),
                ("--tl-fp8-llm-mlp", tl_fp8_llm_mlp),
                ("--tl-fp8-expert-mlp", tl_fp8_expert_mlp),
            )
            if on
        ]
        if compile_model and fused_flags:
            raise NotImplementedError(
                "--compile is not supported together with the fused Triton tiers "
                f"({', '.join(fused_flags)}); drop --compile or the fused flag"
            )
        self.compile_model = bool(compile_model)
        if self.compile_model:
            # TF32 for the remaining fp32 matmuls (action/time projections), same as smolvla.
            torch.set_float32_matmul_precision("high")
            model = self.policy.model
            # (owner, attribute) in inference order; ``compile_region`` reports the
            # attribute path, so the names match the code below by construction.
            self.compiled_regions = [
                compile_region(model, model, "embed_prefix"),
                compile_region(model, model.paligemma_with_expert, "forward"),
                compile_region(model, model, "denoise_step"),
            ]
            log.info(
                "--compile: torch.compiling the inference phases "
                f"({', '.join(self.compiled_regions)}) -- the Euler loop stays eager, "
                "so the step body is compiled once instead of once per step"
            )

        # overlap (default on, graph mode only): fuse the LLM prefill with the first
        # denoising step so the step-0 expert layers run inside the prefill window (step-0
        # layer i only needs prefill layer i's prefix KV). It is a model-level execution
        # mode, not a graph construct, so the flag is pushed to the model; ``--graph`` then
        # captures the fused branch as one graph per shape, bit-exact vs the sequential one.
        self.overlap = bool(overlap) and self.graph_enabled and not self.compile_model
        if self.compile_model and self.graph_enabled and bool(overlap):
            log.info(
                "--compile: overlap is incompatible (the compiled phases are the sequential "
                "entry points); overlap auto-disabled"
            )
        # The model owns the interleave; set it before the warm-up so the overlap branch's
        # side stream / events are created outside the capture (not capture-safe).
        self.policy.model.overlap = bool(self.overlap)
        # Per-shape graph state, keyed by ``(n_cams, lang_len)``, so ``--graph-cameras 2,3``
        # pre-captures those shapes. ``--pad-free`` graphs are bucketed by the number of
        # *real* cameras sent (each count skips trailing empty slots); the non-pad-free graph
        # uses every image slot under one key. Registry shared with the other backends.
        self._sess = CaptureSession(self.device)
        self._graph_runner = GraphRunner(self._capture_graph, self.device, label="pi05 graph", logger=log)

        if warmup:
            self._warmup()
        if self.graph_enabled:
            keys = self._graph_keys(graph_cameras)
            if not self._graph_runner.precapture(keys):
                log.warning("CUDA graph capture failed for all requested keys; falling back to eager")
                self.graph_enabled = False
                self.overlap = False
                self.policy.model.use_static = False
                self.policy.model.overlap = False
                self._graph_runner.clear()
        if profile:
            try:
                self.profile()
            except Exception as e:  # noqa: BLE001 - never fail startup on a diagnostic
                log.warning(f"profile failed: {e}")

    def supports_rtc(self) -> bool:
        """pi0.5 implements Real-Time Chunking guidance (no extra weights)."""
        return True

    def _build_rtc_config(self, override: dict | None) -> RTCConfig | None:
        """Merge the checkpoint ``rtc_config`` block with CLI overrides.

        Returns ``None`` when RTC is not requested or is explicitly disabled.
        """
        raw: dict = {}
        if self.config.rtc_config:
            raw.update(self.config.rtc_config)
        if override:
            raw.update(override)
        if not raw:
            return None
        cfg = RTCConfig.from_dict(raw)
        if cfg is None or not cfg.enabled:
            return None
        return cfg

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "pi05",
            "cameras": [key[len("observation.images.") :] for key in self.config.real_camera_keys],
            "resize": [self.config.image_resolution[1], self.config.image_resolution[0]],  # (width, height)
            "pad_mode": "center",  # pi0.5 resizes with centered padding (openpi)
            "action_dim": int(self.config.action_feature_shape[0]),
            "chunk_size": int(self.config.chunk_size),
            "rtc": self.rtc_processor is not None,
        }

    def _batch_from_frame(self, frame: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.preprocessor(frame)

    def _prepare_noise(self, noise: torch.Tensor | None) -> torch.Tensor | None:
        if noise is None:
            return None
        if noise.device != self.device:
            noise = noise.to(self.device)
        if noise.shape[-1] != self.config.max_action_dim:
            noise = F.pad(noise, (0, self.config.max_action_dim - noise.shape[-1]))
        return noise

    def _prepare_leftover(self, prev_chunk_left_over) -> torch.Tensor:
        """Convert a client-supplied RTC prefix to a model-space tensor.

        Accepts ``(T_prev, action_dim)`` or ``(B, T_prev, action_dim)`` numpy / list /
        tensor in *model space* (normalized actions, real action dim).
        """
        if isinstance(prev_chunk_left_over, torch.Tensor):
            t = prev_chunk_left_over.detach().float()
        else:
            t = torch.as_tensor(np.asarray(prev_chunk_left_over, dtype=np.float32))
        if t.ndim == 3:
            t = t.squeeze(0)
        return t.to(self.device)

    # CUDA graph path: one graph per (camera count, language length), captured lazily on
    # first sight. ``--pad-free`` buckets by real camera count and bucketed language length;
    # the non-pad-free graph uses every image slot plus the full language block.
    # The packing rule lives in the model (``packed_lang_len`` / ``pack_language``) and eager
    # uses the same packing -- that is what makes the ``--pad-free`` graph bit-identical to
    # eager. Both paths derive the slot count from ``PI05Config.live_camera_keys(batch)``.
    _LANG_BUCKET = 16

    def _slot_count(self, batch: dict) -> int:
        """Image slots this request uses: all slots, or only the live cameras with --pad-free."""
        if self.pad_free:
            return len(self.config.live_camera_keys(batch))
        return len(self.config.image_features)

    def _lang_bucket(self, n_live: int) -> int:
        """Round the live language-token count up to the bucket grid (model-owned rule)."""
        return self.policy.model.packed_lang_len(n_live)

    def _graph_keys(self, graph_cameras: tuple[int, ...] | None) -> list[tuple[int, int]]:
        """``(camera count, language length)`` keys to pre-capture at startup.

        The full language length is the safe fallback; compacted buckets are captured
        lazily on first sight.
        """
        full_lang = self.config.tokenizer_max_length
        if not self.pad_free:
            return [(len(self.config.image_features), full_lang)]
        max_cams = len(self.config.real_camera_keys)
        # Default to 2 cameras (clamped to what the checkpoint has); other counts
        # are opt-in via --graph-cameras (or captured lazily on first sight).
        cams = camera_counts(graph_cameras, default=2, max_images=max_cams)
        return [(c, full_lang) for c in cams]

    def _capture_graph(self, key, request=None) -> dict[str, Any]:
        """Capture the model's whole inference core as ONE graph for ``key = (n_cams, lang_len)``.

        The captured unit is the model's own entry point, ``PI05FlowMatching.sample_actions``
        (vision -> prefix embedding -> LLM prefill -> denoising loop, with the step-0 expert
        layers interleaved when ``overlap`` is set): the model's side-stream fork/join becomes
        graph edges, so one graph holds that fusion. The prefix length ``256*n_cams + lang_len``
        is baked into the static buffers; the denoising noise is generated *outside* the graph.

        Every tensor the graph reads must stay reachable for its whole lifetime: the inputs
        live in the returned ``statics`` and the rest in the model's persistent state (graph
        pool buffers, ``_overlap_stream`` / ``_overlap_events``). Nothing may live only in a
        local closure here.
        """
        n_cams, lang_len = int(key[0]), int(key[1])
        model = self.policy.model
        dev = self.device
        model.use_static = True
        model._n_img_slots = n_cams  # the key's first element: the live slot count
        model._lang_len = lang_len
        model._precompute_constants(dev)
        n_feat = n_cams
        prefix_len = 256 * n_feat + lang_len
        h, w = self.config.image_resolution

        images = [torch.zeros(1, 3, h, w, dtype=torch.float32, device=dev) for _ in range(n_feat)]
        img_masks = [torch.ones(1, dtype=torch.bool, device=dev) for _ in range(n_feat)]
        tokens = torch.zeros(1, lang_len, dtype=torch.int64, device=dev)
        masks = torch.ones(1, lang_len, dtype=torch.bool, device=dev)
        noise = torch.zeros(1, self.config.chunk_size, self.config.max_action_dim, dtype=torch.float32, device=dev)
        static: dict[str, Any] = {
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "noise": noise,
        }
        # ``_precompute_constants`` REPLACES these model-level tensors on every capture, so
        # a later key would free the memory this key's graphs read. Snapshot them here:
        # the runner must keep this alive (a captured graph may only read tensors it holds).
        keepalive = model.constant_tensors()

        def run():
            # Keep the captured output (pool-backed, stable address across replays);
            # read it via ``static["out"].clone()``.
            static["out"] = model.sample_actions(images, img_masks, tokens, masks, noise=noise)

        # Prime the fused Triton kernels on the default stream BEFORE capture: the packed-qkv
        # / MLP kernels only run with the ``use_static`` AdaRMS modulations, so ``_warmup``
        # never compiles them, and compiling inside the capture warmup can bake a stale kernel.
        if self.tl_fused_expert or self.tl_fp8_expert_mlp:
            model.sample_actions(images, img_masks, tokens, masks, noise=noise)
            torch.cuda.synchronize()

        log.info(
            f"capturing one CUDA graph for {n_cams} camera(s) "
            f"({'step-0 fused into the LLM prefill' if self.overlap else 'sequential'}, "
            f"lang={lang_len}, prefix_len={prefix_len}, steps={self.config.num_steps})"
        )
        # Warm up on a side stream so cuBLAS/cuDNN heuristics and workspaces settle before
        # capture (which bakes the kernel choices) and the overlap side stream / events
        # already exist.
        graph = self._sess.prime_and_capture(run)
        self._sess.sync()
        log.info(
            f"CUDA graph captured ({n_cams} cameras, lang={lang_len}, "
            f"prefix_len={prefix_len}, steps={self.config.num_steps}): {_GRAPH_COMPONENTS}"
        )
        return {"graph": graph, "statics": static, "keepalive": keepalive}

    def _run_graph(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None) -> torch.Tensor:
        """Copy the batch into the static buffers, replay the graph, return the chunk.

        Returns a clone of the static output buffer, sliced to the real action dim, so
        later replays cannot overwrite it.
        """
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        if self.pad_free:
            # Image slots (the eager path calls the same ``live_camera_keys``) and
            # language packing must match eager, or the prefix length differs.
            image_keys = self.config.live_camera_keys(batch)
            n_cams = len(image_keys)
            tokens, masks = self.policy.model.pack_language(tokens, masks)
            lang_len = int(tokens.shape[1])
        else:
            image_keys = list(self.config.image_features)
            n_cams = len(image_keys)
            lang_len = self.config.tokenizer_max_length
        key = (n_cams, lang_len)  # both elements come from the packing rules above
        entry = self._graph_runner.entry(key)  # lazy capture for a key startup missed
        static = entry["statics"]
        images, img_masks = self.policy._preprocess_images(batch, image_features=image_keys)

        with self._graph_runner.copy_inputs():
            for i, (img, mask) in enumerate(zip(images, img_masks, strict=True)):
                static["images"][i].copy_(img)
                static["img_masks"][i].copy_(mask)
            static["tokens"].copy_(tokens)
            static["masks"].copy_(masks)
            if noise is not None:
                static["noise"].copy_(noise)
            else:
                # Fresh Gaussian noise per request, generated OUTSIDE the graph (replays
                # reuse the RNG state baked at capture); uses the engine RNG (--seed)
                # when configured.
                torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=tuple(static["noise"].shape),
                    dtype=torch.float32,
                    device=self.device,
                    generator=self._noise_gen,
                    out=static["noise"],
                )
        self._graph_runner.wait_inputs()
        # The whole core is in one graph, the overlap fusion included (its side-stream
        # fork/join became graph edges at capture time): the replay is identical either way.
        entry["graph"].replay()
        out = static["out"].clone()
        return out[:, :, : self.config.action_feature_shape[0]]

    # ``-> Any``: the chunk is a ``np.ndarray`` unless ``return_normalized`` is set, in which case
    # it is the ``(unnormalized, normalized)`` pair documented below. The base class declares the
    # ``np.ndarray`` form and ``tybok/worker.py`` unpacks the pair only on the path that asks for
    # it; fastwam annotates its own variant return the same way.
    def predict_action_chunk(
        self,
        frame: dict[str, Any],
        noise: torch.Tensor | None = None,
        *,
        prev_chunk_left_over=None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        return_normalized: bool = False,
    ) -> Any:
        """Return the full action chunk ``(chunk_size, action_dim)``, unnormalized.

        Args:
            noise: float tensor over the *real* action space ``(1, chunk_size,
                action_dim)``; zero-padded to ``max_action_dim`` when necessary.
            prev_chunk_left_over: RTC only -- unexecuted tail of the previous chunk in
                model space, ``(T_prev, action_dim)`` or ``(B, T_prev, action_dim)``;
                the denoising trajectory is guided towards that prefix.
            inference_delay: RTC only -- prefix timesteps used for guidance (default 0).
            execution_horizon: RTC only -- prefix-weight horizon (default: config value).
            return_normalized: also return the normalized chunk.

        Returns:
            ``(chunk_size, action_dim)``, or an ``(unnormalized, normalized)`` tuple.
        """
        batch = self._batch_from_frame(frame)
        noise = self._prepare_noise(noise)

        guidance = self.rtc_processor is not None and prev_chunk_left_over is not None
        if guidance:
            if inference_delay is None:
                inference_delay = 0
            left = self._prepare_leftover(prev_chunk_left_over)
            # RTC guidance runs under ``torch.enable_grad()``, and ``inference_mode`` cannot
            # be re-enabled mid-graph: use ``no_grad`` instead (identical numerics).
            with torch.no_grad():
                actions = self.policy.predict_action_chunk(
                    batch,
                    noise=noise,
                    prev_chunk_left_over=left,
                    inference_delay=inference_delay,
                    execution_horizon=execution_horizon,
                )
        elif self.graph_enabled and self._graph_runner:
            actions = self._run_graph(batch, noise)
        else:
            with torch.inference_mode():
                actions = self.policy.predict_action_chunk(batch, noise=noise)

        chunk = self.postprocessor(actions)
        if return_normalized:
            return chunk.cpu().numpy()[0], actions.detach().cpu().numpy()[0]
        return chunk.cpu().numpy()[0]

    @torch.inference_mode()
    def select_action(self, frame: dict[str, Any], noise: torch.Tensor | None = None) -> np.ndarray:
        """Return a single action (action_dim,), unnormalized (queue-based)."""
        if self.rtc_processor is not None:
            raise ValueError("RTC is not supported for select_action, use predict_action_chunk")
        batch = self._batch_from_frame(frame)
        noise = self._prepare_noise(noise)
        if not self._action_queue:
            if self.graph_enabled and self._graph_runner:
                actions = self._run_graph(batch, noise)[:, : self.config.n_action_steps]
            else:
                actions = self.policy.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        action = self._action_queue.popleft()
        action = self.postprocessor(action)
        return action.cpu().numpy()[0]

    @torch.inference_mode()
    def validate(self, reference_dir: str) -> dict[str, float]:
        """Compare engine outputs against reference dumps; returns max-abs / mean-abs errors."""

        def load(name: str):
            path = os.path.join(reference_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"reference dump {path} missing")
            return torch.load(path, weights_only=False)

        frame = load("frame.pt")
        ref_batch = load("batch.pt")
        ref_prefix = load("prefix.pt")
        ref_kv = load("prefix_kv.pt")
        ref_denoise = load("denoise.pt")
        ref_chunk = load("chunk.pt")

        errors: dict[str, float] = {}

        def max_err(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
            a = a.detach().float().cpu()
            b = b.detach().float().cpu()
            if a.shape != b.shape:
                raise AssertionError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            diff = (a - b).abs()
            errors[name + ".max"] = float(diff.max())
            errors[name + ".mean"] = float(diff.mean())

        # 1. preprocessed batch
        batch = self._batch_from_frame(frame)
        for key in ["observation.state", "observation.language.tokens", "observation.language.attention_mask"] + [
            k for k in ref_batch if k.startswith("observation.images.")
        ]:
            if key not in batch or key not in ref_batch:
                continue
            max_err(f"batch.{key}", batch[key], ref_batch[key])

        # 2. embed_prefix
        # Steps 2-4 need the FULL prefix even when the graph was captured with a reduced
        # one (--pad-free skips empty camera slots), so disable the static path here.
        _use_static = self.policy.model.use_static
        self.policy.model.use_static = False
        images, img_masks = self.policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        embs, pad_masks, att_masks = self.policy.model.embed_prefix(images, img_masks, tokens, masks)
        max_err("prefix.embs", embs, ref_prefix["embs"])
        max_err("prefix.pad_masks", pad_masks, ref_prefix["pad_masks"])
        max_err("prefix.att_masks", att_masks, ref_prefix["att_masks"])

        # 3. prefix KV cache
        from .models.flow_matching import make_att_2d_masks, prepare_attention_masks_4d

        prefix_att_2d = make_att_2d_masks(pad_masks, att_masks)
        prefix_pos_ids = torch.cumsum(pad_masks, dim=1) - 1
        _, kv = self.policy.model.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(prefix_att_2d),
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        assert kv is not None
        for i in range(len(kv)):
            ref = ref_kv[f"layer{i}"]
            max_err(f"prefix_kv.layer{i}.keys", kv[i][0], ref["keys"])
            max_err(f"prefix_kv.layer{i}.values", kv[i][1], ref["values"])

        # 4. single denoise step
        bsize = tokens.shape[0]
        x_t = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, device=self.device)
        timestep = torch.tensor([0.5], dtype=torch.float32, device=self.device)
        if self.tl_fused_expert or self.tl_fp8_expert_mlp:
            # Exercise the triton kernels: that branch needs the per-norm AdaRMS
            # modulations (normally precomputed by the static path), built here with the
            # exact eager ops so the rest of the step stays bit-identical to eager.
            model = self.policy.model
            emb = create_sinusoidal_pos_embedding(
                timestep,
                model.action_in_proj.out_features,
                min_period=model.config.min_period,
                max_period=model.config.max_period,
                device=self.device,
            ).type(dtype=torch.float32)
            cond = model.compute_adarms_cond(emb)
            v_t = model.denoise_step(
                prefix_pad_masks=pad_masks,
                past_key_values=kv,
                x_t=x_t,
                timestep=timestep,
                time_emb=emb,
                adarms_mods=model.compute_adarms_mods(cond),
                adarms_cond=cond,
            )
        else:
            v_t = self.policy.model.denoise_step(
                prefix_pad_masks=pad_masks, past_key_values=kv, x_t=x_t, timestep=timestep
            )
        max_err("denoise.v_t", v_t, ref_denoise["v_t"])

        # 5. full chunk (zero noise, deterministic) -- through the graph when enabled so
        # validate also pins the captured path. With the fused expert tiers, enable the
        # static precompute so every layer takes the triton branch.
        self.policy.model.use_static = _use_static
        noise = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, device=self.device)
        if self.graph_enabled and self._graph_runner:
            chunk = self._run_graph(batch, noise)
        elif self.tl_fused_expert or self.tl_fp8_expert_mlp:
            model = self.policy.model
            model.use_static = True
            model._precompute_constants(self.device)
            chunk = self.policy.predict_action_chunk(batch, noise=noise)
        else:
            chunk = self.policy.predict_action_chunk(batch, noise=noise)
        max_err("chunk", chunk, ref_chunk["chunk"])
        chunk_post = self.postprocessor(chunk)
        max_err("chunk_post", chunk_post, ref_chunk["chunk_post"])

        return errors


def print_validation_report(errors: dict[str, float]) -> None:
    worst = sorted(errors.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'component':<48} {'max-abs':>12} {'mean-abs':>12}")
    for name, err in worst:
        kind = "max" if name.endswith(".max") else "mean"
        comp = name[: -len(kind) - 1]
        if kind == "max":
            print(f"{comp:<48} {err:>12.6g}")
        else:
            print(f"{'':<48} {err:>12.6g}  ({comp})")
