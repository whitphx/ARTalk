#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Streaming renderer for motion → RGB.

Streaming counterpart to ``ARTAvatarInferEngine.rendering``. The
one-shot path collects all frames in a list and muxes them with audio
into an MP4; this class renders motion frames as they arrive and
returns RGB tensors. Audio/video muxing is left to the caller's
transport layer.

Two modes mirror the one-shot paths:

* ``'mesh'`` — FLAME mesh render at 512x512.
* ``'gagavatar'`` — GAGAvatar Gaussian-splat render of a tracked
  real avatar.

The renderer accepts the FLAME / GAGAvatar components as constructor
arguments rather than loading them itself, so callers can share an
``ARTAvatarInferEngine``'s already-loaded modules without paying the
init cost twice.

See ``docs/realtime.md`` for the design rationale, including why the
existing one-shot ``rendering()`` is kept untouched and why "raw
motion params output" is not a renderer mode (callers that want to
skip server-side rendering simply consume motion frames before this
stage).
"""

import torch


def _module_device(module):
    # FLAMEModel holds only registered buffers, no trainable parameters.
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    raise ValueError(
        f"cannot infer device from {type(module).__name__}; pass device="
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
        output_uint8=False,
    ):
        if mode == "mesh":
            if basic_vae is None or flame_model is None or mesh_renderer is None:
                raise ValueError(
                    "mode='mesh' requires basic_vae, flame_model, mesh_renderer"
                )
            if device is None:
                device = _module_device(flame_model)
            if shape_code is None:
                shape_code = torch.zeros(1, 300, device=device)
            else:
                if shape_code.dim() != 2 or shape_code.shape[0] != 1:
                    raise ValueError(
                        f"shape_code must be (1, 300), got {tuple(shape_code.shape)}"
                    )
                shape_code = shape_code.to(device)
            self._render = self._render_mesh
            self._render_batch = self._render_mesh_batch
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
                device = _module_device(gagavatar_flame)
            gagavatar.set_avatar_id(shape_id)
            self._render = self._render_gagavatar
            self._render_batch = self._render_gagavatar_batch
            self._gaga = gagavatar
            self._gaga_flame = gagavatar_flame
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self._device = device
        # Scale/clamp/permute to HWC uint8 on the GPU before the
        # device-to-host copy: 4x less PCIe traffic than float32 and no
        # CPU-side conversion. Batch outputs are then (T, H, W, 3) uint8
        # instead of (T, 3, H, W) float.
        self._output_uint8 = bool(output_uint8)

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
    def render_batch(self, motion_frames):
        """Render a motion batch in one model invocation.

        ``motion_frames``: 2-D tensor of shape (T, motion_dim). Returns
        a (T, 3, H, W) ``torch.float32`` tensor on CPU with values in
        [0, 1], or (T, H, W, 3) ``torch.uint8`` with ``output_uint8``.

        In ``'gagavatar'`` mode this requires a batch-capable
        ``gagavatar`` implementation (the packaged ``gagavatar``
        runtime); the copy vendored under ``artalk/GAGAvatar`` builds
        forward batches one frame at a time — use ``render_frame`` /
        ``feed`` with it.
        """
        if motion_frames.dim() != 2:
            raise ValueError(
                f"motion_frames must be 2-D, got shape {tuple(motion_frames.shape)}"
            )
        return self._render_batch(motion_frames.to(self._device))

    @torch.no_grad()
    def feed(self, motion_frames):
        """Iterate over (T, motion_dim) and yield per-frame RGB tensors."""
        if motion_frames.dim() != 2:
            raise ValueError(
                f"motion_frames must be 2-D, got shape {tuple(motion_frames.shape)}"
            )
        for i in range(motion_frames.shape[0]):
            yield self.render_frame(motion_frames[i])

    def _batch_to_output(self, rgb_batch):
        if not self._output_uint8:
            return rgb_batch.cpu()
        return (
            (rgb_batch * 255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
        )

    def _render_mesh(self, motion_frame):
        verts = self._mesh_basic_vae.get_flame_verts(
            self._mesh_flame,
            self._mesh_shape_code,
            motion_frame[None],
            with_global=True,
        )
        rgb = self._mesh_renderer(verts)[0]
        return rgb.cpu()[0] / 255.0

    def _render_mesh_batch(self, motion_frames):
        shape_code = self._mesh_shape_code.expand(motion_frames.shape[0], -1)
        verts = self._mesh_basic_vae.get_flame_verts(
            self._mesh_flame,
            shape_code,
            motion_frames,
            with_global=True,
        )
        rgb = self._mesh_renderer(verts)[0] / 255.0
        return self._batch_to_output(rgb)

    def _render_gagavatar(self, motion_frame):
        batch = self._gaga.build_forward_batch(motion_frame[None], self._gaga_flame)
        rgb = self._gaga.forward_expression(batch)
        return rgb.cpu()[0]

    def _render_gagavatar_batch(self, motion_frames):
        batch = self._gaga.build_forward_batch(motion_frames, self._gaga_flame)
        rgb = self._gaga.forward_expression(batch)
        return self._batch_to_output(rgb)
