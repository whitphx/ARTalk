#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Streaming per-frame renderer for motion → RGB.

Streaming counterpart to ``ARTAvatarInferEngine.rendering``. The
one-shot path collects all frames in a list and muxes them with audio
into an MP4; this class renders a single motion frame at a time and
returns the RGB tensor. Audio/video muxing is left to the transport
layer (e.g. a WebRTC sink in Phase 4).

Two modes mirror the one-shot paths:

* ``'mesh'`` — FLAME mesh render at 512x512.
* ``'gagavatar'`` — GAGAvatar Gaussian-splat render of a tracked
  real avatar.

The renderer accepts the FLAME / GAGAvatar components as constructor
arguments rather than loading them itself, so callers can share an
``ARTAvatarInferEngine``'s already-loaded modules without paying the
init cost twice.

See ``docs/realtime.md`` (Phase 3) for the design rationale, including
why the existing one-shot ``rendering()`` is kept untouched and why
"raw motion params output" is not a renderer mode (callers that want
to skip server-side rendering simply consume motion frames before
this stage).
"""

import time

import torch

from .metrics import (
    active_pipeline_metrics,
    observe_pipeline_duration_if_active,
    pipeline_metrics_scope_if_active,
)


class StreamingRenderer:
    def __init__(
        self,
        mode,
        *,
        basic_vae=None,
        flame_model=None,
        mesh_renderer=None,
        shape_code=None,
        gagavatar=None,
        gagavatar_flame=None,
        shape_id=None,
        device=None,
        stage_sync=True,
    ):
        if mode == "mesh":
            if basic_vae is None or flame_model is None or mesh_renderer is None:
                raise ValueError(
                    "mode='mesh' requires basic_vae, flame_model, mesh_renderer"
                )
            if device is None:
                device = next(flame_model.parameters()).device
            if shape_code is None:
                shape_code = torch.zeros(1, 300, device=device)
            else:
                if shape_code.dim() != 2 or shape_code.shape[0] != 1:
                    raise ValueError(
                        f"shape_code must be (1, 300), got {tuple(shape_code.shape)}"
                    )
                shape_code = shape_code.to(device)
            self._render = self._render_mesh
            self._mesh_basic_vae = basic_vae
            self._mesh_flame = flame_model
            self._mesh_renderer = mesh_renderer
            self._mesh_shape_code = shape_code
        elif mode == "gagavatar":
            if gagavatar is None or gagavatar_flame is None or shape_id is None:
                raise ValueError(
                    "mode='gagavatar' requires gagavatar, gagavatar_flame, shape_id"
                )
            if device is None:
                device = next(gagavatar_flame.parameters()).device
            gagavatar.set_avatar_id(shape_id)
            self._render = self._render_gagavatar
            self._gaga = gagavatar
            self._gaga_flame = gagavatar_flame
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self._device = device
        # Stage-boundary torch.cuda.synchronize() makes the per-stage timings
        # in the *_profile paths attributable, but serializes GPU work that
        # could otherwise overlap. Disable to measure production behavior;
        # per-stage timings then only cover kernel launch, not execution.
        self._stage_sync = bool(stage_sync)

    @property
    def device(self):
        return self._device

    @torch.no_grad()
    def render_frame(self, motion_frame):
        """Render one motion frame.

        ``motion_frame``: 1-D tensor of shape (motion_dim,) — typically
        (106,). Returns a (3, H, W) ``torch.float32`` tensor on CPU
        with values in [0, 1].
        """
        if motion_frame.dim() != 1:
            raise ValueError(
                f"motion_frame must be 1-D, got shape {tuple(motion_frame.shape)}"
            )
        return self._render(motion_frame.to(self._device))

    @torch.no_grad()
    def render_frame_profile(self, motion_frame):
        """Render one frame and return ``(rgb, timings)``.

        Timings are wall-clock seconds. CUDA devices are synchronized at
        stage boundaries so GPU work is attributed to the stage that queued it.
        This is intentionally a development/profiling path; ``render_frame``
        remains the lower-overhead production path.
        """
        if motion_frame.dim() != 1:
            raise ValueError(
                f"motion_frame must be 1-D, got shape {tuple(motion_frame.shape)}"
            )
        with pipeline_metrics_scope_if_active("renderer"):
            with observe_pipeline_duration_if_active("motion_frame_to_device"):
                motion_frame = motion_frame.to(self._device)
            if self.mode == "gagavatar":
                return self._render_gagavatar_profile(motion_frame)
            return self._render_mesh_profile(motion_frame)

    @torch.no_grad()
    def render_batch_profile(self, motion_frames):
        """Render a motion batch and return ``(rgb_batch, timings)``.

        ``motion_frames``: 2-D tensor of shape (T, motion_dim). Returns
        a (T, 3, H, W) ``torch.float32`` tensor on CPU with values in
        [0, 1]. Timings are batch totals, not per-frame durations.
        """
        if motion_frames.dim() != 2:
            raise ValueError(
                f"motion_frames must be 2-D, got shape {tuple(motion_frames.shape)}"
            )
        with pipeline_metrics_scope_if_active("renderer"):
            with observe_pipeline_duration_if_active("motion_batch_to_device"):
                motion_frames = motion_frames.to(self._device)
            if self.mode == "gagavatar":
                return self._render_gagavatar_batch_profile(motion_frames)
            return self._render_mesh_batch_profile(motion_frames)

    @torch.no_grad()
    def feed(self, motion_frames):
        """Iterate over (T, motion_dim) and yield per-frame RGB tensors."""
        if motion_frames.dim() != 2:
            raise ValueError(
                f"motion_frames must be 2-D, got shape {tuple(motion_frames.shape)}"
            )
        for i in range(motion_frames.shape[0]):
            yield self.render_frame(motion_frames[i])

    def _render_mesh(self, motion_frame):
        verts = self._mesh_basic_vae.get_flame_verts(
            self._mesh_flame,
            self._mesh_shape_code,
            motion_frame[None],
            with_global=True,
        )
        rgb = self._mesh_renderer(verts)[0]
        return rgb.cpu()[0] / 255.0

    def _render_mesh_profile(self, motion_frame):
        with pipeline_metrics_scope_if_active("mesh"):
            timings = {}
            t0 = time.perf_counter()
            verts = self._mesh_basic_vae.get_flame_verts(
                self._mesh_flame,
                self._mesh_shape_code,
                motion_frame[None],
                with_global=True,
            )
            self._sync_if_cuda()
            timings["avatar_prepare_frame"] = time.perf_counter() - t0
            self._observe_active("flame_vertices", timings["avatar_prepare_frame"])

            t0 = time.perf_counter()
            rgb = self._mesh_renderer(verts)[0]
            self._sync_if_cuda()
            timings["avatar_forward_model"] = time.perf_counter() - t0
            self._observe_active("pytorch3d_forward", timings["avatar_forward_model"])

            t0 = time.perf_counter()
            rgb = rgb.cpu()[0] / 255.0
            timings["avatar_gpu_to_cpu_copy"] = time.perf_counter() - t0
            self._observe_active("gpu_to_cpu", timings["avatar_gpu_to_cpu_copy"])
            self._record_cuda_memory()
            return rgb, timings

    def _render_mesh_batch_profile(self, motion_frames):
        with pipeline_metrics_scope_if_active("mesh"):
            timings = {}
            t0 = time.perf_counter()
            shape_code = self._mesh_shape_code.expand(motion_frames.shape[0], -1)
            verts = self._mesh_basic_vae.get_flame_verts(
                self._mesh_flame,
                shape_code,
                motion_frames,
                with_global=True,
            )
            self._sync_if_cuda()
            timings["avatar_prepare_batch"] = time.perf_counter() - t0
            self._observe_active("flame_vertices_batch", timings["avatar_prepare_batch"])

            t0 = time.perf_counter()
            rgb = self._mesh_renderer(verts)[0] / 255.0
            self._sync_if_cuda()
            timings["avatar_forward_batch"] = time.perf_counter() - t0
            self._observe_active("pytorch3d_forward_batch", timings["avatar_forward_batch"])

            t0 = time.perf_counter()
            rgb = rgb.cpu()
            timings["avatar_gpu_to_cpu_batch"] = time.perf_counter() - t0
            self._observe_active("gpu_to_cpu_batch", timings["avatar_gpu_to_cpu_batch"])
            self._record_cuda_memory()
            return rgb, timings

    def _render_gagavatar(self, motion_frame):
        batch = self._gaga.build_forward_batch(motion_frame[None], self._gaga_flame)
        rgb = self._gaga.forward_expression(batch)
        return rgb.cpu()[0]

    def _render_gagavatar_profile(self, motion_frame):
        with pipeline_metrics_scope_if_active("gagavatar"):
            timings = {}
            t0 = time.perf_counter()
            batch = self._gaga.build_forward_batch(motion_frame[None], self._gaga_flame)
            self._sync_if_cuda()
            timings["avatar_prepare_frame"] = time.perf_counter() - t0
            self._observe_active("prepare_frame", timings["avatar_prepare_frame"])

            t0 = time.perf_counter()
            rgb = self._gaga.forward_expression(batch)
            self._sync_if_cuda()
            timings["avatar_forward_model"] = time.perf_counter() - t0
            self._observe_active("forward_model", timings["avatar_forward_model"])

            t0 = time.perf_counter()
            rgb = rgb.cpu()[0]
            timings["avatar_gpu_to_cpu_copy"] = time.perf_counter() - t0
            self._observe_active("gpu_to_cpu", timings["avatar_gpu_to_cpu_copy"])
            self._record_cuda_memory()
            return rgb, timings

    def _render_gagavatar_batch_profile(self, motion_frames):
        with pipeline_metrics_scope_if_active("gagavatar"):
            timings = {}
            t0 = time.perf_counter()
            batch = self._gaga.build_forward_batch(motion_frames, self._gaga_flame)
            self._sync_if_cuda()
            timings["avatar_prepare_batch"] = time.perf_counter() - t0
            self._observe_active("prepare_batch", timings["avatar_prepare_batch"])

            t0 = time.perf_counter()
            rgb = self._gaga.forward_expression(batch)
            self._sync_if_cuda()
            timings["avatar_forward_batch"] = time.perf_counter() - t0
            self._observe_active("forward_batch", timings["avatar_forward_batch"])

            t0 = time.perf_counter()
            rgb = rgb.cpu()
            timings["avatar_gpu_to_cpu_batch"] = time.perf_counter() - t0
            self._observe_active("gpu_to_cpu_batch", timings["avatar_gpu_to_cpu_batch"])
            self._record_cuda_memory()
            return rgb, timings

    def _sync_if_cuda(self):
        if not self._stage_sync:
            return
        device = torch.device(self._device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _observe_active(key, elapsed_s):
        metrics = active_pipeline_metrics()
        if metrics is not None:
            metrics.observe_ms(key, elapsed_s)

    def _record_cuda_memory(self):
        metrics = active_pipeline_metrics()
        device = torch.device(self._device)
        if metrics is None or device.type != "cuda":
            return
        metrics.set(
            "cuda_memory_allocated_mb",
            torch.cuda.memory_allocated(device) / (1024 * 1024),
        )
        metrics.set(
            "cuda_memory_reserved_mb",
            torch.cuda.memory_reserved(device) / (1024 * 1024),
        )
        metrics.observe_max(
            "cuda_max_memory_allocated_mb",
            torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        )
