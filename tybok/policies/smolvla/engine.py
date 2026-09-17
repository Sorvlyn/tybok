"""SmolVLA policy engine (registered as ``model_type="smolvla"``).

Deployment entry point for the ``smolvla`` backend: loads the checkpoint and exposes
:meth:`SmolVLAEngine.select_action` / :meth:`SmolVLAEngine.predict_action_chunk`.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import deque
from typing import Any

import numpy as np
import torch

from ...base import PolicyEngine, compile_region
from ...graph import (
    CaptureSession,
    GraphRunner,
    camera_counts,
)
from ...registry import register
from .config import SmolVLAConfig
from .models.policy import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    SmolVLAPolicy,
    load_model,
)
from .preprocess import PostProcessor, PreProcessor, load_normalizer, load_unnormalizer
from .tokenizer import TaskTokenizer

log = logging.getLogger("tybok.smolvla")

# The model components the captured graph bakes in, as ``<role>=<module path>``; the single
# graph of ``VLAFlowMatching.sample_actions`` covers all three.
_GRAPH_COMPONENTS = "vision encoder=vlm.model.vision_model, llm backbone=vlm.model.text_model, action expert=lm_expert"


@register("smolvla")
class SmolVLAEngine(PolicyEngine):
    # Flags owned by other backends (pi05 / fastwam). The worker forwards them with these
    # inert defaults; any other value means the flag was requested and is rejected.
    _UNSUPPORTED_FLAGS = {
        # pi05-only
        "tl_fused_vit": False,
        "tl_llm_flash_attn": False,
        "tl_fp8_llm_mlp": False,
        "tl_fp8_expert_mlp": False,
        "vit_mlp_dtype": None,
        "skip_empty_images": False,
        "pad_free": False,
        # fastwam-only
        "pack_qkv": False,
        "action_context_cache": True,
        "cache_prompt": True,
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
        cache_expert_prefix_kv: bool = True,
        profile: bool = False,
        overlap: bool = True,
        camera_alias: dict[str, str] | None = None,
        tl_llm_fused_attn: bool = False,
        tl_vit_oproj: bool = False,
        tl_fused_expert: bool = False,
    ):
        super().__init__(checkpoint_dir, device=device)
        if sampler not in ("euler", "heun"):
            raise ValueError(f"unknown sampler {sampler!r} (expected 'euler' or 'heun')")
        self.config = SmolVLAConfig.from_pretrained(checkpoint_dir)
        if num_steps is not None:
            # Read at build time (static caches, capture), so set before warmup; fewer steps
            # change the output.
            self.config.num_steps = int(num_steps)
        # Read at model build time; set before constructing the policy.
        self.config.sampler = sampler
        self.config.cache_expert_prefix_kv = bool(cache_expert_prefix_kv)
        vlm_dir = self.config.vlm_dir or self.config.vlm_model_name
        if self.config.vlm_config is None or not vlm_dir:
            raise ValueError(
                f"Could not locate the VLM backbone config for {self.config.vlm_model_name!r}; "
                "check `vlm_model_name` in the checkpoint config.json"
            )
        self.tokenizer = TaskTokenizer(vlm_dir, max_length=self.config.tokenizer_max_length)

        # bf16 build: action/state and the expert cross-attn k/v projections stay fp32.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.policy = SmolVLAPolicy(self.config, tokenizer=self.tokenizer)
        finally:
            torch.set_default_dtype(default_dtype)

        skipped = load_model(self.policy, checkpoint_dir)
        if skipped:
            log.info(f"skipped {len(skipped)} unmatched weights (e.g. {skipped[0]})")

        self.policy.to(self.device).eval()

        norm_map = self.config.normalization_mapping
        # --camera-alias: rename robot cameras to the checkpoint's image slots (camera names
        # or full ``observation.images.*`` keys); applied only when the target slot is missing.
        self.camera_alias: dict[str, str] | None = None
        if camera_alias:
            img_feats = self.config.image_features
            full: dict[str, str] = {}
            for src, dst in camera_alias.items():
                src_key = src if src.startswith("observation.images.") else f"observation.images.{src}"
                dst_key = dst if dst.startswith("observation.images.") else f"observation.images.{dst}"
                if dst_key not in img_feats:
                    raise ValueError(
                        f"camera_alias target {dst!r} is not an expected image feature "
                        f"(available: {[k[len('observation.images.') :] for k in img_feats]})"
                    )
                if src_key in img_feats:
                    raise ValueError(
                        f"camera_alias source {src!r} is already an expected image feature "
                        f"({[k[len('observation.images.') :] for k in img_feats]})"
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

        # CUDA-graph state, keyed by ``(n_cams,)``; ``--graph-cameras 2,3`` pre-captures the
        # 2- and 3-image shapes (see ``tybok.graph``).
        self.graph_requested = bool(graph)
        self.graph_enabled = self.graph_requested and str(self.device).startswith("cuda")
        self._sess = CaptureSession(self.device)
        self._graph_runner = GraphRunner(self._capture_graph, self.device, label="smolvla graph", logger=log)
        # A request is one replay, so no per-camera event is needed.
        self._graph_fallback_warned = False
        self._action_queue: deque[torch.Tensor] = deque(maxlen=self.config.n_action_steps)

        # ``--compile`` and the fused Triton tiers are mutually exclusive: inductor precompiles
        # the same kernels with configurations the eager launcher never uses, which fails to build.
        fused_flags = [
            name
            for name, on in (
                ("--tl-llm-fused-attn", tl_llm_fused_attn),
                ("--tl-vit-oproj", tl_vit_oproj),
                ("--tl-fused-expert", tl_fused_expert),
            )
            if on
        ]
        if compile_model and fused_flags:
            raise NotImplementedError(
                "--compile is not supported together with the fused Triton tiers "
                f"({', '.join(fused_flags)}); drop --compile or the fused flag"
            )
        # ``--compile`` compiles the inference *phases* the selected path calls, not
        # ``sample_actions``: the denoise step body is compiled once, not once per step.
        # Regions: eager path -> ``embed_prefix`` + ``vlm_with_expert.forward`` + ``denoise_step``;
        # ``--graph`` capture -> ``denoise_step`` only (a compiled prefill invalidates the capture).
        # The step body it compiles is the sequential core's, so overlap is turned off below.
        self.compile_model = bool(compile_model)
        # KV-cache setting taken out for the compile; restored at the end of ``__init__``.
        self._compile_kv_cache_restore = False
        if compile_model:
            torch.set_float32_matmul_precision("high")
            model = self.policy.model
            # (model, attribute) pairs in inference order.
            regions: list[tuple[torch.nn.Module, str]] = []
            if not self.graph_enabled:
                regions += [(model, "embed_prefix"), (model.vlm_with_expert, "forward")]
            regions.append((model, "denoise_step"))
            # Per-request mutable state (fresh KVCache, bumped ``fill_count``) that dynamo guards
            # on: compile the stateless form instead, which re-projects the same k/v inline.
            self._compile_kv_cache_restore = bool(model.cache_expert_prefix_kv)
            if self._compile_kv_cache_restore:
                log.info(
                    "--compile: disabling cache_expert_prefix_kv (expert prefix-KV projection "
                    "cache) for the compile -- it is per-request mutable state (new KVCache "
                    "object + fill_count + new k/v tensors per request) and dynamo guards on it, "
                    "so the traced region would re-specialise whenever it changes; the compiled "
                    "region projects the same k/v inline instead"
                )
                model.cache_expert_prefix_kv = False
            self.compiled_regions = [compile_region(model, owner, attr) for owner, attr in regions]
            log.info(
                "--compile: torch.compiling the inference phases "
                f"({', '.join(self.compiled_regions)})"
                + (
                    " -- the prefix embed / LLM prefill stay eager inside the captured core"
                    if self.graph_enabled
                    else " -- the denoise loop stays eager, so the step body is compiled once instead of once per step"
                )
            )

        # overlap (graph mode only; ``--no-overlap`` turns it off): fuse the LLM prefill and the
        # first denoise step so step 0 hides inside the prefill window. Bit-exact; CUDA + Euler.
        self.overlap = (
            bool(overlap)
            and not compile_model
            and str(self.device).startswith("cuda")
            and self.config.sampler == "euler"
            and self.graph_requested
        )
        if compile_model and self.graph_enabled:
            log.info(
                "--compile + --graph: the compiled entry is the denoise step body; the prefix "
                "embed / LLM prefill stay eager -- a compiled prefill invalidates the CUDA-graph "
                "capture"
            )
        if compile_model and self.graph_enabled and bool(overlap):
            log.info(
                "--compile: overlap is incompatible (the compiled entry points are the "
                "sequential ones; the fused branch is a different entry); overlap auto-disabled"
            )

        # --tl-fused-expert: fused Triton GQA kernels for the expert attention chains, with the
        # post-attention RMSNorm fused into the MLP gate/up chain. Drift tier; set before warmup.
        self.tl_fused_expert = bool(tl_fused_expert)
        if self.tl_fused_expert:
            self.policy.model.vlm_with_expert.triton_attn = True
            for layer in self.policy.model.vlm_with_expert.lm_expert.layers:
                layer.mlp.fused_norm_gate_up = True

        # --tl-llm-fused-attn: Triton LLM prefill attention (RMSNorm + q/k/v + RoPE + KV write,
        # then GQA flash); o_proj / residual / MLP stay eager. Drift tier; set before warmup.
        self.tl_llm_fused_attn = bool(tl_llm_fused_attn)
        if self.tl_llm_fused_attn:
            self.policy.model.vlm_with_expert.fused_prefill_attn = True

        # --tl-vit-oproj: vision ``out_proj`` as one Triton kernel over the SDPA output layout;
        # q/k/v are packed into one cuBLAS GEMM, which that kernel requires. Drift tier.
        self.tl_vit_oproj = bool(tl_vit_oproj)
        if self.tl_vit_oproj:
            vision = self.policy.model.vlm_with_expert.vlm.model.vision_model
            for layer in vision.encoder.layers:
                layer.self_attn.fused_qkv = True
                layer.self_attn.fused_out_proj = True
                layer.self_attn.build_qkv()

        # --seed: engine-owned RNG for the denoising noise (``None`` uses the global RNG).
        self.seed = seed
        self._noise_gen: torch.Generator | None = None
        if seed is not None:
            self._noise_gen = torch.Generator(device=self.device).manual_seed(int(seed))
            self.policy.model.noise_generator = self._noise_gen

        if self.graph_enabled:
            # Persistent model caches: stable addresses at capture, no allocation inside it.
            self.policy.model.use_static_cache = True
            max_prefix = self._prefix_len_for(len(self.config.image_features))
            self.policy.model.ensure_kv_cache(max_prefix, batch_size=1, device=self.device)
            # The model's step-0 fusion branch owns the side stream / events, so the flag must be
            # set before the warm-up: resource creation is not capture-safe.
            self.policy.model.overlap = bool(self.overlap)
            # Event for the double-buffered noise: the *next* request's noise is prefetched on
            # ``GraphRunner.copy_inputs``' stream while the replay runs (see ``_run_graph``).
            self._noise_event = torch.cuda.Event()  # prefetched next noise ready
            self._noise_staging = torch.zeros(
                1,
                self.config.chunk_size,
                self.config.max_action_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self._noise_ready = False

        if warmup:
            self._warmup()
        if self.graph_enabled:
            # Pre-capture one graph per camera count so no request pays capture latency
            # mid-session; defaults to the configured count.
            max_cams = len(self.config.image_features)
            counts = camera_counts(graph_cameras, default=max_cams, max_images=max_cams)
            if not self._graph_runner.precapture([(n,) for n in counts]):
                log.warning("CUDA graph capture failed for all requested camera counts; falling back to eager mode")
                self.graph_enabled = False
                self.overlap = False
                self._graph_runner.clear()
                self.policy.model.use_static_cache = False

        # --profile: one-shot startup latency report for the worker logs.
        if profile:
            try:
                self.profile()
            except Exception as e:  # noqa: BLE001 - never fail startup on a diagnostic
                log.warning(f"profile failed: {e}")

        # Restore the setting the compile took out: the runtime path runs with what the caller
        # asked for, and the first request may re-specialise the compiled regions once.
        if self._compile_kv_cache_restore:
            self.policy.model.cache_expert_prefix_kv = True
            log.info(
                "--compile: cache_expert_prefix_kv restored to True for the runtime path "
                "(re-projecting inline stays in the compiled regions; the first request may "
                "re-specialise them once)"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "smolvla",
            "cameras": [key[len("observation.images.") :] for key in self.config.image_features],
            "resize": list(self.config.resize_imgs_with_padding),  # (width, height)
            "action_dim": int(self.config.action_feature_shape[0]),
            "chunk_size": int(self.config.chunk_size),
        }

    def _batch_from_frame(self, frame: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.preprocessor(frame)

    @torch.inference_mode()
    def predict_action_chunk(self, frame: dict[str, Any], noise: torch.Tensor | None = None) -> np.ndarray:
        """Return the full chunk ``(chunk_size, action_dim)``, unnormalized.

        Args:
            noise: float tensor over the *real* action space
                ``(1, chunk_size, action_dim)``; zero-padded to ``max_action_dim``.
        """
        batch = self._batch_from_frame(frame)
        noise = self._prepare_noise(noise)
        if self._graph_runner:
            try:
                actions = self._run_graph(batch, noise)
            except ValueError as e:
                # shapes outside the captured set (e.g. camera count) fall back to eager here
                self._warn_graph_fallback(e)
                actions = self.policy.predict_action_chunk(batch, noise=noise)
        else:
            actions = self.policy.predict_action_chunk(batch, noise=noise)
        actions = actions[:, :, : self.config.action_feature_shape[0]]
        chunk = self.postprocessor(actions)
        return chunk.cpu().numpy()[0]

    @torch.inference_mode()
    def select_action(self, frame: dict[str, Any], noise: torch.Tensor | None = None) -> np.ndarray:
        """Return a single action ``(action_dim,)``, unnormalized.

        In CUDA-graph mode the queue lives on the engine, so a chunk is inferred only
        when it is empty.
        """
        batch = self._batch_from_frame(frame)
        noise = self._prepare_noise(noise)
        if self._graph_runner:
            if not self._action_queue:
                try:
                    actions = self._run_graph(batch, noise)  # [1, chunk, max_action_dim]
                except ValueError as e:
                    self._warn_graph_fallback(e)
                    actions = self.policy.predict_action_chunk(batch, noise=noise)
                actions = actions[:, :, : self.config.action_feature_shape[0]]
                self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
            action = self._action_queue.popleft()  # [1, action_dim]
        else:
            action = self.policy.select_action(batch, noise=noise)
        action = self.postprocessor(action)
        return action.cpu().numpy()[0]

    def _prepare_noise(self, noise: torch.Tensor | None) -> torch.Tensor | None:
        if noise is None:
            return None
        if noise.device != self.device:
            noise = noise.to(self.device)
        if noise.shape[-1] != self.config.max_action_dim:
            import torch.nn.functional as F  # noqa: N812

            noise = F.pad(noise, (0, self.config.max_action_dim - noise.shape[-1]))
        return noise

    def _prefix_len_for(self, n_cams: int) -> int:
        """Return the prefix length in tokens for a request with ``n_cams`` images."""
        vlm = self.policy.model.vlm_with_expert
        extra = 2 if self.config.add_image_special_tokens else 0
        return n_cams * (vlm.config.image_seq_len + extra) + self.config.tokenizer_max_length + 1

    def _capture_graph(self, key, request=None) -> dict[str, Any]:
        """Capture the model's whole inference core as one graph for ``key = (n_cams,)``.

        Args:
            key: ``(n_cams,)``; the prefix length is baked into the static buffers, so there is
                one graph per camera count.
            request: unused; part of the ``GraphRunner`` capture signature.

        Returns:
            ``{"graph": ..., "statics": ...}``. Captured kernels are the eager kernels, so this
            must stay bit-exact vs eager. Every tensor the graph reads must stay reachable for
            its whole lifetime -- inputs in ``statics``, reuse buffers and the side stream /
            events in the model -- never only in a local closure.
        """
        n_cams = int(key[0])
        spec = self.describe()
        img_h, img_w = spec["resize"][1], spec["resize"][0]
        prefix_len = self._prefix_len_for(n_cams)

        # Static input buffers, overwritten before every replay. All-ones masks keep capture-time
        # values in bounds (valid bucketized position ids).
        static: dict[str, Any] = {
            "images": [torch.zeros(1, 3, img_h, img_w, dtype=torch.float32, device=self.device) for _ in range(n_cams)],
            "img_masks": [torch.ones(1, dtype=torch.bool, device=self.device) for _ in range(n_cams)],
            "lang_tokens": torch.zeros(1, self.config.tokenizer_max_length, dtype=torch.long, device=self.device),
            "lang_masks": torch.ones(1, self.config.tokenizer_max_length, dtype=torch.bool, device=self.device),
            "state": torch.zeros(1, self.config.max_state_dim, dtype=torch.float32, device=self.device),
            "noise": torch.zeros(
                1,
                self.config.chunk_size,
                self.config.max_action_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            "out": torch.zeros(
                1,
                self.config.chunk_size,
                self.config.max_action_dim,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        model = self.policy.model

        def core() -> None:
            static["out"].copy_(
                model.sample_actions(
                    static["images"],
                    static["img_masks"],
                    static["lang_tokens"],
                    static["lang_masks"],
                    static["state"],
                    noise=static["noise"],
                )
            )

        log.info(
            f"capturing one CUDA graph for {n_cams} camera(s) "
            f"({'step-0 fused into the LLM prefill' if self.overlap else 'sequential'}, "
            f"prefix_len={prefix_len}, steps={self.config.num_steps})"
        )
        graph = self._sess.prime_and_capture(core)
        self._sess.sync()
        log.info(f"CUDA graph captured ({n_cams} cameras, prefix_len={prefix_len}): {_GRAPH_COMPONENTS}")
        return {"graph": graph, "statics": static}

    def _run_graph(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None) -> torch.Tensor:
        """Copy the batch into the static buffers, replay the graph, return the chunk.

        Args:
            batch: preprocessed batch, already on ``self.device``.
            noise: ``(1, chunk_size, max_action_dim)`` or None to sample fresh noise.

        Returns:
            ``(1, chunk_size, max_action_dim)`` copy of the static output buffer, so later
            replays cannot change it.

        Input copies and this replay's noise are issued on the copy stream owned by
        ``GraphRunner.copy_inputs``; ``wait_inputs`` joins them before the replay. The next
        request's noise is prefetched on that stream after the replay, one generator draw per
        inference, keeping the RNG sequence identical to the eager path.
        """
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        n_cams = len(images)
        entry = self._graph_runner.entry((n_cams,))
        static = entry["statics"]
        noise_implicit = noise is None

        # (1) input copies + this replay's noise on the copy stream.
        with self._graph_runner.copy_inputs() as copy_stream:
            for i, (img, mask) in enumerate(zip(images, img_masks, strict=True)):
                static["images"][i].copy_(img)
                static["img_masks"][i].copy_(mask)
            static["lang_tokens"].copy_(batch[OBS_LANGUAGE_TOKENS])
            static["lang_masks"].copy_(batch[OBS_LANGUAGE_ATTENTION_MASK])
            static["state"].copy_(state)

            if noise_implicit:
                # Noise must be generated OUTSIDE the graph: a replay reuses the RNG state baked
                # at capture and would otherwise return the same noise every time.
                if self._noise_ready:
                    copy_stream.wait_event(self._noise_event)
                    static["noise"].copy_(self._noise_staging)
                    self._noise_ready = False
                else:
                    torch.normal(
                        mean=0.0,
                        std=1.0,
                        size=tuple(static["noise"].shape),
                        dtype=torch.float32,
                        device=self.device,
                        generator=self._noise_gen,
                        out=static["noise"],
                    )
            else:
                static["noise"].copy_(noise)
        self._graph_runner.wait_inputs()

        # (2) replay: the one graph covers the whole core, the step-0 fusion included.
        entry["graph"].replay()
        out = static["out"]
        assert out is not None, "graph output missing"

        # (3) prefetch the next request's noise, concurrent with the replay; only after
        # implicit-noise calls, so the generator advances once per inference, in order.
        if noise_implicit:
            with torch.cuda.stream(copy_stream):
                torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=tuple(self._noise_staging.shape),
                    dtype=torch.float32,
                    device=self.device,
                    generator=self._noise_gen,
                    out=self._noise_staging,
                )
                self._noise_event.record(copy_stream)
            self._noise_ready = True

        # static buffer: copy it out before the next replay overwrites it
        return out.clone()

    def _warn_graph_fallback(self, exc: ValueError) -> None:
        """Warn once when a request cannot use the captured graph (e.g. missing camera)."""
        if not getattr(self, "_graph_fallback_warned", False):
            log.warning(f"CUDA-graph replay skipped ({exc}); falling back to eager for this request")
            self._graph_fallback_warned = True

    def _profile_header_extra(self) -> str:
        """smolvla's fusion tier, appended to the shared profile header line."""
        return f"fused-attn/oproj/expert={self.tl_llm_fused_attn}/{self.tl_vit_oproj}/{self.tl_fused_expert}"

    def _profile_phases(
        self, iters: int = 5, warmup: int = 3, cameras: int | None = None, seed: int = 0
    ) -> dict[str, float]:
        """Return per-phase wall time (ms/iter) of the eager components via CUDA events.

        Phase names match the model primitives (and the ``--compile`` regions). On a
        graph-enabled engine they time the eager components the captured graph is built from.
        ``cameras=None`` uses all configured cameras. Falls back to ``perf_counter`` on CPU.
        """
        from .models.flow_matching import make_att_2d_masks
        from .models.policy import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = self._batch_from_frame(self._profile_frame(cameras, seed))
        model = self.policy.model
        cfg = self.config
        noise = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, device=self.device)

        images = img_masks = state = None
        embs = pad_masks = att_masks = None
        att_2d_masks = None
        pos = None
        kv_prefix = None
        x_final = None

        def prepare_images():
            nonlocal images, img_masks, state
            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)

        def embed_prefix():
            nonlocal embs, pad_masks, att_masks
            embs, pad_masks, att_masks = model.embed_prefix(
                images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
            )

        def vlm_forward():
            nonlocal att_2d_masks, pos, kv_prefix
            assert pad_masks is not None and att_masks is not None, "embed_prefix must run first"
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            pos = torch.cumsum(pad_masks, dim=1) - 1
            _, kv_prefix = model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=pos,
                past_key_values=None,  # fresh cache, like the eager non-static path
                inputs_embeds=[embs, None],
                use_cache=cfg.use_cache,
                fill_kv_cache=True,
            )

        def denoise_loop():
            nonlocal x_final
            # reuses the prefix KV built above, so this row times the denoise loop only.
            assert pad_masks is not None and kv_prefix is not None, "vlm_with_expert.forward must run first"
            x_final = model._denoise_loop(pad_masks, kv_prefix, noise, None, pad_masks.shape[0], self.device)

        def postprocess():
            assert x_final is not None, "_denoise_loop must run first"
            self.postprocessor(x_final[:, :, : cfg.action_feature_shape[0]]).cpu()

        steps = {
            "prepare_images": prepare_images,
            "embed_prefix": embed_prefix,
            "vlm_with_expert.forward": vlm_forward,
            "_denoise_loop": denoise_loop,
            "postprocessor": postprocess,
        }
        phases = tuple(steps)

        def measure(fn) -> float:
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                stop.record()
                torch.cuda.synchronize()
                return start.elapsed_time(stop)
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1e3

        for _ in range(warmup):
            with torch.inference_mode():
                for fn in steps.values():
                    fn()
        acc: dict[str, list[float]] = {name: [] for name in phases}
        for _ in range(iters):
            with torch.inference_mode():
                for name in phases:
                    acc[name].append(measure(steps[name]))
        return {name: statistics.mean(v) for name, v in acc.items()}

    @torch.inference_mode()
    def validate(self, reference_dir: str) -> dict[str, float]:
        """Compare engine outputs against reference dumps; per-component max/mean-abs errors.

        Runs under ``inference_mode`` so the static-path KV buffers (allocated during warmup,
        also inference mode) stay writable from the eager branches.
        """

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
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        embs, pad_masks, att_masks = self.policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        max_err("prefix.embs", embs, ref_prefix["embs"])
        max_err("prefix.pad_masks", pad_masks, ref_prefix["pad_masks"])
        max_err("prefix.att_masks", att_masks, ref_prefix["att_masks"])

        # 3. prefix KV cache
        from .models.flow_matching import make_att_2d_masks

        prefix_att_2d = make_att_2d_masks(pad_masks, att_masks)
        prefix_pos_ids = torch.cumsum(pad_masks, dim=1) - 1
        _, kv = self.policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        assert kv is not None
        for i in range(self.config.num_vlm_layers):
            ref = ref_kv[f"layer{i}"]
            max_err(f"prefix_kv.layer{i}.keys", kv[i]["key_states"], ref["keys"])
            max_err(f"prefix_kv.layer{i}.values", kv[i]["value_states"], ref["values"])

        # 4. single denoise step
        bsize = state.shape[0]
        x_t = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, device=self.device)
        timestep = torch.tensor([0.5], dtype=torch.float32, device=self.device)
        v_t = self.policy.model.denoise_step(prefix_pad_masks=pad_masks, past_key_values=kv, x_t=x_t, timestep=timestep)
        max_err("denoise.v_t", v_t, ref_denoise["v_t"])

        # 5. full chunk (zero noise, deterministic); through the CUDA graph when enabled
        noise = torch.zeros(bsize, self.config.chunk_size, self.config.max_action_dim, device=self.device)
        if self._graph_runner:
            try:
                chunk = self._run_graph(batch, noise)[:, :, : self.config.action_feature_shape[0]]
            except ValueError as e:
                # e.g. the reference frame carries only a subset of the cameras
                self._warn_graph_fallback(e)
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
