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
FPS = 25
RENDER_RES = (512, 512)
AUDIO_OUT_PTIME = 0.020
AUDIO_OUT_SAMPLES_PER_FRAME = int(SAMPLE_RATE * AUDIO_OUT_PTIME)


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
        # Output audio buffer: int16 mono samples at SAMPLE_RATE,
        # produced by the worker as it consumes input audio into the
        # streamer, drained by audio_source_callback at real-time
        # pacing so video and audio share the same ~4 s latency.
        self._audio_out_buffer = np.zeros(0, dtype=np.int16)
        self._audio_out_lock = threading.Lock()
        # Worker-only staging for input audio that the streamer has
        # accumulated but not yet "consumed" (i.e. produced motion
        # for). When motion fires we move the matching prefix into
        # _audio_out_buffer.
        self._pending_audio_for_output: list[np.ndarray] = []
        self._stop_event = threading.Event()
        self._dbg_calls = 0
        h, w = RENDER_RES
        self._initial_placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        self._placeholder = self._initial_placeholder
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
            self._placeholder = arr
        except queue.Empty:
            arr = self._placeholder
        return av.VideoFrame.from_ndarray(arr, format="rgb24")

    def audio_source_callback(
        self, pts: int, time_base: fractions.Fraction
    ) -> av.AudioFrame:
        """Synchronous audio producer paired with the video output so
        both share the model's ~4 s chunk latency. Drains
        ``AUDIO_OUT_SAMPLES_PER_FRAME`` samples from
        ``_audio_out_buffer`` per call, padding with silence when the
        buffer is short (the gap before the first motion fires, or any
        underrun) so the outbound track keeps timestamping at the
        configured ptime.
        """
        n = AUDIO_OUT_SAMPLES_PER_FRAME
        with self._audio_out_lock:
            available = self._audio_out_buffer.size
            if available >= n:
                samples = self._audio_out_buffer[:n].copy()
                self._audio_out_buffer = self._audio_out_buffer[n:]
            elif available > 0:
                samples = np.zeros(n, dtype=np.int16)
                samples[:available] = self._audio_out_buffer
                self._audio_out_buffer = np.zeros(0, dtype=np.int16)
            else:
                samples = np.zeros(n, dtype=np.int16)
        # AudioFrame.from_ndarray for s16 mono expects shape (1, N).
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        return frame

    def stop(self):
        self._stop_event.set()
        self._placeholder = self._initial_placeholder
        with self._audio_out_lock:
            self._audio_out_buffer = np.zeros(0, dtype=np.int16)
        self._pending_audio_for_output = []

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
        # Stage the input audio so we can hand the matching prefix to
        # the output side once the streamer actually consumes it (i.e.
        # when motion frames are produced).
        self._pending_audio_for_output.append(samples_int16.copy())

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
                "buf_after=%d motion=%s audio_q=%d video_q=%d "
                "audio_out_buf=%d",
                self._dbg_calls,
                samples_int16.size,
                buf_before,
                buf_after,
                tuple(motion.shape),
                self._audio_in_queue.qsize(),
                self.video_queue.qsize(),
                self._audio_out_buffer.size,
            )
        if motion.shape[0] == 0:
            return

        # Couple the matching audio with the produced motion: each
        # frame corresponds to SAMPLE_RATE / FPS = 640 input samples,
        # so 100 frames consume 64000 samples (= 4 s @ 16 kHz).
        n_audio_samples = motion.shape[0] * SAMPLE_RATE // FPS
        all_pending = np.concatenate(self._pending_audio_for_output)
        emitted_audio = all_pending[:n_audio_samples]
        leftover = all_pending[n_audio_samples:]
        self._pending_audio_for_output = (
            [leftover] if leftover.size > 0 else []
        )
        with self._audio_out_lock:
            self._audio_out_buffer = np.concatenate(
                [self._audio_out_buffer, emitted_audio]
            )

        logger.warning(
            "[ARTalkPipeline] motion produced call#%d motion=%s "
            "audio_emitted=%d",
            self._dbg_calls,
            tuple(motion.shape),
            emitted_audio.size,
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
            "[ARTalkPipeline] frames rendered call#%d video_q=%d "
            "audio_out_buf=%d",
            self._dbg_calls,
            self.video_queue.qsize(),
            self._audio_out_buffer.size,
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
