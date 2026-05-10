#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Per-session streaming pipeline for the streamlit-webrtc demo.

Wires the streaming pieces from Phases 1-3 (``ARTalkStreamer`` →
``CausalSavgolSmoother`` → ``StreamingRenderer`` mesh mode) so that
audio frames pushed in via streamlit-webrtc's ``audio_frame_callback``
flow through to RGB video frames pulled out by
``video_source_callback`` (the callback streamlit-webrtc's
``create_video_source_track`` invokes at the configured fps).

The audio callback returns immediately after enqueueing samples; a
dedicated daemon worker thread drains the queue and runs the heavy
streamer / smoother / renderer chain. This is required because every
4 seconds of audio triggers ~100 frames of mesh rendering, which on
a typical GPU runs for seconds — doing that work inline in the audio
callback would stall streamlit-webrtc's audio path and the browser
would see the session "freeze".

Single-session, single-GPU MVP — see ``docs/realtime.md`` Phase 4.
"""

import fractions
import logging
import queue
import threading

import av
import numpy as np
import torch

from .rendering import StreamingRenderer
from .streaming import ARTalkStreamer, CausalSavgolSmoother


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
RENDER_RES = (512, 512)


class ARTalkPipeline:
    """Per-session pipeline holding the streaming model state."""

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
        self._audio_in_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._dbg_calls = 0
        h, w = RENDER_RES
        self._placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ARTalkPipelineWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def push_audio_frame(self, frame: av.AudioFrame):
        """Fast: resample, enqueue, return. The heavy lift happens in the worker."""
        resampled = self._resampler.resample(frame)
        # av.AudioResampler.resample returns a list in modern PyAV.
        frames = resampled if isinstance(resampled, list) else [resampled]
        for rf in frames:
            arr = rf.to_ndarray()
            if arr.ndim == 2:
                # mono after layout="mono", first row is the channel.
                arr = arr[0]
            if arr.size > 0:
                self._audio_in_queue.put(arr)

    def video_source_callback(
        self, pts: int, time_base: fractions.Fraction
    ) -> av.VideoFrame:
        """Synchronous frame producer for streamlit-webrtc's
        ``create_video_source_track``. Drains one frame from
        ``video_queue`` if available, falls back to a black placeholder
        otherwise — the call must return promptly so the outbound
        track keeps firing at its configured fps.
        """
        try:
            arr = self.video_queue.get_nowait()
        except queue.Empty:
            arr = self._placeholder
        return av.VideoFrame.from_ndarray(arr, format="rgb24")

    def stop(self):
        self._stop_event.set()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                samples_int16 = self._audio_in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process_audio_chunk(samples_int16)
            except Exception:
                logger.exception("[ARTalkPipeline] worker error")

    def _process_audio_chunk(self, samples_int16: np.ndarray):
        samples_t = torch.from_numpy(
            samples_int16.astype(np.float32) / 32768.0
        ).to(self._device)
        buf_before = self._streamer._audio_buffer.shape[0]
        motion = self._streamer.feed(samples_t)
        buf_after = self._streamer._audio_buffer.shape[0]
        self._dbg_calls += 1
        if self._dbg_calls <= 3 or self._dbg_calls % 50 == 0:
            logger.warning(
                "[ARTalkPipeline] call#%d in_samples=%d buf_before=%d "
                "buf_after=%d motion=%s audio_q=%d video_q=%d",
                self._dbg_calls,
                samples_int16.size,
                buf_before,
                buf_after,
                tuple(motion.shape),
                self._audio_in_queue.qsize(),
                self.video_queue.qsize(),
            )
        if motion.shape[0] == 0:
            return
        logger.warning(
            "[ARTalkPipeline] motion produced call#%d motion=%s",
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
        logger.warning(
            "[ARTalkPipeline] frames rendered call#%d video_q=%d",
            self._dbg_calls,
            self.video_queue.qsize(),
        )

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
