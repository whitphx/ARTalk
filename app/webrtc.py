#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""FastRTC handler for ARTalk realtime streaming.

Wires the streaming inference pipeline (Phases 1-3) onto a FastRTC
``AsyncAudioVideoStreamHandler``: browser microphone audio →
``ARTalkStreamer`` → ``CausalSavgolSmoother`` → ``StreamingRenderer``
(mesh mode) → browser video track. The ~4 second base latency is
structural to the released model; see ``docs/realtime.md`` Phase 4.

Inference is serialized through a single per-connection worker task
that drains an audio input queue and offloads each step to a thread
via ``asyncio.to_thread``, so streamer / smoother / renderer state
stays consistent while the event loop remains responsive for the
inbound audio track.
"""

import asyncio

import numpy as np
import torch
from fastrtc import AsyncAudioVideoStreamHandler

from .rendering import StreamingRenderer
from .streaming import ARTalkStreamer, CausalSavgolSmoother


SAMPLE_RATE = 16000


class ARTalkHandler(AsyncAudioVideoStreamHandler):
    """Mesh-mode ARTalk streaming over WebRTC.

    Construct once at server startup with shared (across connections)
    model + FLAME modules; FastRTC clones per browser connection via
    ``copy()`` and per-connection streaming state is set up in
    ``start_up()``.
    """

    def __init__(
        self,
        *,
        model,
        flame_model,
        mesh_renderer,
        device,
        style_motion=None,
    ):
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            output_sample_rate=SAMPLE_RATE,
            expected_layout="mono",
        )
        self._model = model
        self._flame_model = flame_model
        self._mesh_renderer = mesh_renderer
        self._device = device
        self._style_motion = style_motion
        # Per-instance state. FastRTC may call video_emit on the
        # template / pre-startup clone before start_up runs, so the
        # queues must exist by the time __init__ returns to avoid
        # AttributeError. Streamer / smoother / renderer are lazy so
        # the template doesn't allocate GPU state at server startup.
        self._streamer = None
        self._smoother = None
        self._renderer = None
        self._video_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._audio_in_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = None

    def copy(self):
        return ARTalkHandler(
            model=self._model,
            flame_model=self._flame_model,
            mesh_renderer=self._mesh_renderer,
            device=self._device,
            style_motion=self._style_motion,
        )

    async def start_up(self):
        await self._ensure_worker()

    async def receive(self, frame):
        await self._ensure_worker()
        _, samples = frame
        if samples.size == 0:
            return
        # FastRTC delivers int16 mono shape (1, N) at input_sample_rate.
        samples_f32 = samples.astype(np.float32) / 32768.0
        samples_t = torch.from_numpy(samples_f32[0]).to(self._device)
        await self._audio_in_queue.put(samples_t)

    async def emit(self):
        # MVP: no outbound audio. Users can monitor their own mic via
        # system sidetone for lip reference; mixing audio at a 4 s lag
        # to match the avatar video is left for a future iteration.
        return None

    async def video_receive(self, frame):
        # Inbound video is unused.
        return

    async def video_emit(self):
        await self._ensure_worker()
        # FastRTC drives this at the outbound video track's pacing.
        # The 100-frame burst produced per 4 s audio chunk is buffered
        # in _video_queue, so successive video_emit calls drain the
        # queue at whatever rate FastRTC requests.
        return await self._video_queue.get()

    async def shutdown(self):
        worker = self._worker_task
        if worker and not worker.done():
            worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass

    def _ensure_pipeline(self):
        if self._streamer is None:
            self._streamer = ARTalkStreamer(self._model, style_motion=self._style_motion)
            self._smoother = CausalSavgolSmoother()
            self._renderer = StreamingRenderer(
                mode="mesh",
                basic_vae=self._model.basic_vae,
                flame_model=self._flame_model,
                mesh_renderer=self._mesh_renderer,
                device=self._device,
            )

    async def _ensure_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._ensure_pipeline()
            self._worker_task = asyncio.create_task(self._inference_worker())

    async def _inference_worker(self):
        while True:
            audio = await self._audio_in_queue.get()
            try:
                frames = await asyncio.to_thread(self._inference_step, audio)
            except Exception as exc:
                print(f"[ARTalkHandler] inference error: {exc!r}")
                continue
            for f in frames:
                await self._video_queue.put(f)

    def _inference_step(self, audio):
        motion = self._streamer.feed(audio)
        if motion.shape[0] == 0:
            return []
        smoothed = self._smoother.feed(motion)
        if smoothed.shape[0] == 0:
            return []
        out = []
        for rgb in self._renderer.feed(smoothed):
            out.append(self._to_uint8_hwc(rgb))
        return out

    @staticmethod
    def _to_uint8_hwc(rgb_chw):
        return (
            (rgb_chw * 255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
