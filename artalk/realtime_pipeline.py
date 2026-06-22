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
import time

import av
import numpy as np
import torch

from .rendering import StreamingRenderer
from .streaming import ARTalkStreamer, CausalSavgolSmoother


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FPS = 25
DEFAULT_RENDER_RES = 512
DEFAULT_RENDER_BATCH_SIZE = 4
AUDIO_OUT_PTIME = 0.020
AUDIO_OUT_SAMPLES_PER_FRAME = int(SAMPLE_RATE * AUDIO_OUT_PTIME)


class PipelineMetrics:
    """Thread-safe rolling counters for the development Streamlit app."""

    def __init__(self):
        self._lock = threading.Lock()
        self._created_at = time.perf_counter()
        self._counters: dict[str, int | float] = {}
        self._durations: dict[str, dict[str, float]] = {}

    def inc(self, key: str, value: int | float = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def set(self, key: str, value: int | float | None) -> None:
        if value is None:
            return
        with self._lock:
            self._counters[key] = value

    def set_once(self, key: str, value: int | float) -> bool:
        with self._lock:
            if key in self._counters:
                return False
            self._counters[key] = value
            return True

    def observe_ms(self, key: str, elapsed_s: float) -> None:
        elapsed_ms = elapsed_s * 1000.0
        with self._lock:
            stat = self._durations.setdefault(
                key,
                {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0},
            )
            stat["count"] += 1
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            stat["last_ms"] = elapsed_ms

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            counters = dict(self._counters)
            durations = {key: dict(value) for key, value in self._durations.items()}
        for stat in durations.values():
            count = stat["count"]
            stat["avg_ms"] = stat["total_ms"] / count if count else 0.0
        counters["uptime_s"] = now - self._created_at
        counters["now_s"] = now
        return {"counters": counters, "durations": durations}


class ARTalkPipeline:
    """Per-session pipeline holding the streaming model state."""

    def __init__(
        self,
        *,
        model,
        flame_model,
        mesh_renderer,
        device,
        style_motion=None,
        render_res=DEFAULT_RENDER_RES,
        renderer_mode="mesh",
        gagavatar=None,
        gagavatar_flame=None,
        shape_id=None,
        render_batch_size=DEFAULT_RENDER_BATCH_SIZE,
    ):
        self._device = device
        self._streamer = ARTalkStreamer(model, style_motion=style_motion)
        self._smoother = CausalSavgolSmoother()
        self._renderer = StreamingRenderer(
            mode=renderer_mode,
            basic_vae=model.basic_vae,
            flame_model=flame_model,
            mesh_renderer=mesh_renderer,
            gagavatar=gagavatar,
            gagavatar_flame=gagavatar_flame,
            shape_id=shape_id,
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
        self._render_res = int(render_res)
        self._render_batch_size = max(1, int(render_batch_size))
        self.metrics = PipelineMetrics()
        self.metrics.set("streamer_chunk_samples", self._streamer.patch_audio_length)
        self.metrics.set(
            "streamer_chunk_floor_s",
            self._streamer.patch_audio_length / SAMPLE_RATE,
        )
        self.metrics.set("streamer_frames_per_chunk", self._streamer.frames_per_chunk)
        self.metrics.set("fps", FPS)
        self.metrics.set("audio_out_samples_per_frame", AUDIO_OUT_SAMPLES_PER_FRAME)
        self.metrics.set("render_res", self._render_res)
        self.metrics.set("render_batch_size", self._render_batch_size)
        self._initial_placeholder = np.zeros(
            (self._render_res, self._render_res, 3),
            dtype=np.uint8,
        )
        self._placeholder = self._initial_placeholder
        if self._renderer.mode == "gagavatar":
            self._warm_up_renderer()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ARTalkPipelineWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def push_audio_frame(self, frame: av.AudioFrame):
        """Fast: enqueue the raw frame, return. The worker handles
        resampling to 16 kHz mono and the heavy inference / render."""
        self.metrics.set_once("first_audio_push_s", time.perf_counter())
        self.metrics.inc("audio_frames_pushed")
        self._record_frame_timestamp("input_audio", frame)
        self._audio_in_queue.put(frame)
        self.metrics.set("audio_in_queue_depth", self._audio_in_queue.qsize())

    def video_source_callback(
        self, pts: int, time_base: fractions.Fraction
    ) -> av.VideoFrame:
        """Synchronous frame producer for streamlit-webrtc's
        ``create_video_source_track``. Drains one frame from
        ``video_queue`` if available, falls back to a black placeholder
        otherwise — the call must return promptly so the outbound
        track keeps firing at its configured fps.
        """
        self.metrics.set_once("first_video_callback_s", time.perf_counter())
        self.metrics.inc("video_callbacks")
        self._record_callback_timestamp("video_source", pts, time_base)
        try:
            arr = self.video_queue.get_nowait()
            self._placeholder = arr
            self.metrics.inc("video_frames_served")
        except queue.Empty:
            arr = self._placeholder
            self.metrics.inc("video_placeholder_frames")
        self.metrics.set("video_queue_depth", self.video_queue.qsize())
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

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
        self.metrics.set_once("first_audio_callback_s", time.perf_counter())
        self.metrics.inc("audio_callbacks")
        self._record_callback_timestamp("audio_source", pts, time_base)
        n = AUDIO_OUT_SAMPLES_PER_FRAME
        with self._audio_out_lock:
            available = self._audio_out_buffer.size
            if available >= n:
                samples = self._audio_out_buffer[:n].copy()
                self._audio_out_buffer = self._audio_out_buffer[n:]
                underrun = 0
            elif available > 0:
                samples = np.zeros(n, dtype=np.int16)
                samples[:available] = self._audio_out_buffer
                self._audio_out_buffer = np.zeros(0, dtype=np.int16)
                underrun = n - available
            else:
                samples = np.zeros(n, dtype=np.int16)
                underrun = n
            buffered = self._audio_out_buffer.size
        self.metrics.inc("audio_frames_served")
        if underrun:
            self.metrics.inc("audio_underrun_frames")
            self.metrics.inc("audio_underrun_samples", underrun)
        self.metrics.set("audio_out_buffer_samples", buffered)
        # AudioFrame.from_ndarray for s16 mono expects shape (1, N).
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def _record_frame_timestamp(self, prefix: str, frame) -> None:
        if frame.pts is None:
            return
        self.metrics.set(f"last_{prefix}_pts", frame.pts)
        if frame.time_base is not None:
            time_base = float(frame.time_base)
            self.metrics.set(f"last_{prefix}_time_base", time_base)
            self.metrics.set(f"last_{prefix}_time_s", frame.pts * time_base)

    def _record_callback_timestamp(
        self,
        prefix: str,
        pts: int | None,
        time_base: fractions.Fraction,
    ) -> None:
        if pts is None:
            return
        self.metrics.set(f"last_{prefix}_pts", pts)
        time_base_s = float(time_base)
        self.metrics.set(f"last_{prefix}_time_base", time_base_s)
        self.metrics.set(f"last_{prefix}_time_s", pts * time_base_s)

    def stop(self):
        # Signal the worker first so any frames it produces past this
        # point are doomed and we can drop the queues underneath.
        self._stop_event.set()
        # Reset the visible placeholder so a stop+restart doesn't
        # show the last frame from the previous session.
        self._placeholder = self._initial_placeholder
        # Output side state.
        with self._audio_out_lock:
            self._audio_out_buffer = np.zeros(0, dtype=np.int16)
        self._pending_audio_for_output = []
        # Drain both transit queues. The worker may still write up to
        # one more chunk between its stop_event check and the queue
        # put, but the caller replaces session_state.pipeline after
        # stop() so any tail frames die with the old pipeline. Model
        # state (_streamer / _smoother internal buffers) is left
        # alone because mutating it from another thread can race
        # with an in-flight inference; it's discarded with the
        # pipeline instance instead.
        self._drain_queue(self._audio_in_queue)
        self._drain_queue(self.video_queue)

    @staticmethod
    def _drain_queue(q: queue.Queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _pop_pending_audio_for_output(self, n_samples: int) -> np.ndarray:
        all_pending = (
            np.concatenate(self._pending_audio_for_output)
            if self._pending_audio_for_output
            else np.zeros(0, dtype=np.int16)
        )
        emitted_audio = all_pending[:n_samples]
        leftover = all_pending[n_samples:]
        self._pending_audio_for_output = (
            [leftover] if leftover.size > 0 else []
        )
        self.metrics.set("pending_audio_for_output_samples", leftover.size)
        return emitted_audio

    def _append_output_audio(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self._audio_out_lock:
            self._audio_out_buffer = np.concatenate(
                [self._audio_out_buffer, samples]
            )
            buffered = self._audio_out_buffer.size
        self.metrics.set("audio_out_buffer_samples", buffered)

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _warm_up_renderer(self):
        motion = torch.zeros(
            self._streamer.motion_dim,
            dtype=self._streamer.dtype,
            device=self._streamer.device,
        )
        t0 = time.perf_counter()
        rgb, timings = self._renderer.render_frame_profile(motion)
        for key, elapsed_s in timings.items():
            self.metrics.observe_ms(f"warmup_{key}", elapsed_s)
        convert_t0 = time.perf_counter()
        (
            (rgb * 255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        self.metrics.observe_ms("warmup_rgb_tensor_to_numpy", time.perf_counter() - convert_t0)
        self.metrics.observe_ms("renderer_warmup", time.perf_counter() - t0)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self._audio_in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.metrics.set("audio_in_queue_depth", self._audio_in_queue.qsize())
            try:
                self._process_input_frame(frame)
            except Exception:
                logger.exception("[ARTalkPipeline] worker error")

    def _process_input_frame(self, frame: av.AudioFrame):
        # Resample to 16 kHz mono int16 once; the same chunks are both
        # staged for outbound audio (delayed-emit, paired with rendered
        # video below) and fed to the streamer for motion inference.
        t0 = time.perf_counter()
        resampled = self._resampler.resample(frame)
        self.metrics.observe_ms("resample", time.perf_counter() - t0)
        if not isinstance(resampled, list):
            resampled = [resampled]
        for rf in resampled:
            arr = rf.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            if arr.size == 0:
                continue
            samples = arr.astype(np.int16, copy=True)
            self.metrics.inc("audio_chunks_resampled")
            self.metrics.inc("audio_samples_resampled", samples.size)
            self._pending_audio_for_output.append(samples)
            self._process_audio_chunk(samples)

    def _process_audio_chunk(self, samples_int16: np.ndarray):
        self.metrics.set_once("first_audio_process_s", time.perf_counter())
        samples_t = torch.from_numpy(
            samples_int16.astype(np.float32) / 32768.0
        ).to(self._device)
        buf_before = self._streamer._audio_buffer.shape[0]
        t0 = time.perf_counter()
        motion = self._streamer.feed(samples_t)
        streamer_elapsed = time.perf_counter() - t0
        self.metrics.observe_ms("artalk_streamer_feed", streamer_elapsed)
        buf_after = self._streamer._audio_buffer.shape[0]
        self.metrics.inc("streamer_feed_calls")
        self.metrics.inc("audio_samples_fed_to_streamer", samples_int16.size)
        self.metrics.set("streamer_buffer_samples", buf_after)
        self._dbg_calls += 1
        pending_out_total = sum(p.size for p in self._pending_audio_for_output)
        self.metrics.set("pending_audio_for_output_samples", pending_out_total)
        if self._dbg_calls <= 3 or self._dbg_calls % 50 == 0:
            logger.warning(
                "[ARTalkPipeline] call#%d in_samples=%d buf_before=%d "
                "buf_after=%d motion=%s audio_q=%d video_q=%d "
                "audio_out_buf=%d pending_out=%d",
                self._dbg_calls,
                samples_int16.size,
                buf_before,
                buf_after,
                tuple(motion.shape),
                self._audio_in_queue.qsize(),
                self.video_queue.qsize(),
                self._audio_out_buffer.size,
                pending_out_total,
            )
        if motion.shape[0] == 0:
            return
        now = time.perf_counter()
        if self.metrics.set_once("first_motion_s", now):
            snapshot = self.metrics.snapshot()["counters"]
            first_audio = snapshot.get("first_audio_push_s") or snapshot.get("first_audio_process_s")
            if first_audio is not None:
                self.metrics.set("first_motion_latency_s", now - first_audio)
        self.metrics.inc("motion_chunks_produced")
        self.metrics.inc("motion_frames_produced", motion.shape[0])
        self.metrics.set("last_motion_frames", motion.shape[0])
        self.metrics.set("last_motion_s", now)

        logger.warning(
            "[ARTalkPipeline] motion produced call#%d motion=%s",
            self._dbg_calls,
            tuple(motion.shape),
        )

        t0 = time.perf_counter()
        smoothed = self._smoother.feed(motion)
        self.metrics.observe_ms("smoother_feed", time.perf_counter() - t0)
        self.metrics.inc("smoothed_frames_produced", smoothed.shape[0])
        self.metrics.set("last_smoothed_frames", smoothed.shape[0])
        if smoothed.shape[0] == 0:
            return
        # Couple output audio with frames that will actually be rendered.
        # Each video frame corresponds to SAMPLE_RATE / FPS = 640 samples.
        n_audio_samples_out = smoothed.shape[0] * SAMPLE_RATE // FPS
        emitted_audio = self._pop_pending_audio_for_output(n_audio_samples_out)
        self.metrics.inc("audio_samples_emitted", emitted_audio.size)
        self.metrics.set("last_audio_samples_emitted", emitted_audio.size)
        render_chunk_t0 = time.perf_counter()
        rendered_in_chunk = 0
        audio_cursor = 0
        for start in range(0, smoothed.shape[0], self._render_batch_size):
            motion_batch = smoothed[start : start + self._render_batch_size]
            render_t0 = time.perf_counter()
            rgb_batch, render_timings = self._renderer.render_batch_profile(
                motion_batch
            )
            for key, elapsed_s in render_timings.items():
                self.metrics.observe_ms(key, elapsed_s)
            self.metrics.observe_ms("avatar_render_batch", time.perf_counter() - render_t0)
            self.metrics.inc("render_batches")
            self.metrics.inc("render_batch_frames", motion_batch.shape[0])
            convert_t0 = time.perf_counter()
            arr_batch = (
                (rgb_batch * 255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .numpy()
            )
            self.metrics.observe_ms("rgb_batch_to_numpy", time.perf_counter() - convert_t0)
            for arr in arr_batch:
                self._enqueue_frame(arr)
                self.metrics.inc("rendered_frames")
                rendered_in_chunk += 1
                self.metrics.set("last_rendered_s", time.perf_counter())
            n_batch_audio_samples = arr_batch.shape[0] * SAMPLE_RATE // FPS
            audio_slice = emitted_audio[
                audio_cursor : audio_cursor + n_batch_audio_samples
            ]
            audio_cursor += n_batch_audio_samples
            self._append_output_audio(audio_slice)
        render_chunk_elapsed = time.perf_counter() - render_chunk_t0
        self.metrics.observe_ms("render_chunk_total", render_chunk_elapsed)
        self.metrics.set("last_render_chunk_frames", rendered_in_chunk)
        self.metrics.set("last_render_chunk_s", render_chunk_elapsed)
        if render_chunk_elapsed > 0:
            self.metrics.set(
                "last_render_chunk_fps",
                rendered_in_chunk / render_chunk_elapsed,
            )
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
            self.metrics.set("video_queue_depth", self.video_queue.qsize())
        except queue.Full:
            # Drop the oldest frame and append the new one. Real-time
            # progresses; the gap is preferable to unbounded buildup.
            try:
                self.video_queue.get_nowait()
                self.metrics.inc("video_frames_dropped")
            except queue.Empty:
                pass
            try:
                self.video_queue.put_nowait(arr)
                self.metrics.set("video_queue_depth", self.video_queue.qsize())
            except queue.Full:
                self.metrics.inc("video_frames_drop_failed")
                pass
