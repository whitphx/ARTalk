#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Per-session streaming pipeline for the streamlit-webrtc demo.

Wires the streaming pieces from Phases 1-3 (``ARTalkStreamer`` →
``CausalSavgolSmoother`` → ``StreamingRenderer`` mesh mode) so that
audio frames pushed in via streamlit-webrtc's ``audio_frame_callback``
flow through to RGB video frames pulled out by ``ARTalkVideoTrack``
(an aiortc ``MediaStreamTrack`` subclass) on the outbound side.

Single-session, single-GPU MVP — see ``docs/realtime.md`` Phase 4.
"""

import asyncio
import logging
import queue
import threading

import av
import numpy as np
import torch
from aiortc.mediastreams import MediaStreamTrack

from .rendering import StreamingRenderer
from .streaming import ARTalkStreamer, CausalSavgolSmoother


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
RENDER_RES = (512, 512)


class ARTalkPipeline:
    """Per-session pipeline holding the streaming model state.

    Browser audio is delivered via ``push_audio_frame(av.AudioFrame)``,
    which resamples to 16 kHz mono int16 and feeds the streamer.
    Rendered RGB frames land in ``video_queue`` for ``ARTalkVideoTrack``
    to drain.
    """

    def __init__(self, *, model, flame_model, mesh_renderer, device, style_motion=None):
        self._device = device
        self._streamer = ARTalkStreamer(model, style_motion=style_motion)
        self._smoother = CausalSavgolSmoother()
        self._renderer = StreamingRenderer(
            mode="mesh",
            basic_vae=model.basic_vae,
            flame_model=flame_model,
            mesh_renderer=mesh_renderer,
            device=device,
        )
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )
        self.video_queue: queue.Queue = queue.Queue(maxsize=200)
        self._lock = threading.Lock()
        self._dbg_calls = 0

    def push_audio_frame(self, frame: av.AudioFrame):
        resampled = self._resampler.resample(frame)
        # av.AudioResampler.resample returns a list in modern PyAV.
        frames = resampled if isinstance(resampled, list) else [resampled]
        for rf in frames:
            arr = rf.to_ndarray()
            if arr.ndim == 2:
                # mono after layout="mono", first row is the channel.
                arr = arr[0]
            self._push_audio_samples(arr)

    def _push_audio_samples(self, samples_int16: np.ndarray):
        if samples_int16.size == 0:
            return
        samples_t = torch.from_numpy(
            samples_int16.astype(np.float32) / 32768.0
        ).to(self._device)
        # Lock keeps streamer / smoother / renderer state consistent
        # if the audio callback ever runs on multiple threads.
        with self._lock:
            buf_before = self._streamer._audio_buffer.shape[0]
            motion = self._streamer.feed(samples_t)
            buf_after = self._streamer._audio_buffer.shape[0]
            self._dbg_calls += 1
            if self._dbg_calls <= 3 or self._dbg_calls % 50 == 0:
                logger.warning(
                    "[ARTalkPipeline] call#%d pipeline=%s streamer=%s "
                    "in_samples=%d buf_before=%d buf_after=%d motion=%s",
                    self._dbg_calls,
                    id(self),
                    id(self._streamer),
                    samples_int16.size,
                    buf_before,
                    buf_after,
                    tuple(motion.shape),
                )
            if motion.shape[0] == 0:
                return
            logger.warning(
                "[ARTalkPipeline] motion produced! call#%d motion=%s",
                self._dbg_calls,
                tuple(motion.shape),
            )
            smoothed = self._smoother.feed(motion)
            if smoothed.shape[0] == 0:
                return
            for rgb in self._renderer.feed(smoothed):
                arr = (
                    (rgb * 255.0)
                    .clamp_(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .numpy()
                )
                self._enqueue_frame(arr)

    def _enqueue_frame(self, arr):
        try:
            self.video_queue.put_nowait(arr)
        except queue.Full:
            # Drop the oldest frame and append the new one. Real-time
            # progresses; the gap is preferable to unbounded buildup.
            try:
                self.video_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.video_queue.put_nowait(arr)
            except queue.Full:
                pass


class ARTalkVideoTrack(MediaStreamTrack):
    """aiortc video track that drains ``ARTalkPipeline.video_queue``.

    When no real frame is ready (the first 4 seconds of a session,
    or transient gaps), falls back to a black placeholder so the
    outbound track keeps producing frames at the timestamp pace
    aiortc expects.
    """

    kind = "video"

    def __init__(self, pipeline: ARTalkPipeline, placeholder_hw=RENDER_RES):
        super().__init__()
        self._pipeline = pipeline
        h, w = placeholder_hw
        self._placeholder = np.zeros((h, w, 3), dtype=np.uint8)

    async def recv(self):
        loop = asyncio.get_event_loop()
        try:
            arr = await loop.run_in_executor(
                None,
                lambda: self._pipeline.video_queue.get(timeout=0.04),
            )
        except queue.Empty:
            arr = self._placeholder
        video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
