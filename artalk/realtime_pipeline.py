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
import os
import queue
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import av
import numpy as np
import torch
import torch.profiler

from .metrics import (
    PipelineMetrics,
    current_pipeline_metrics,
    observe_pipeline_duration,
    pipeline_metrics_context,
    pipeline_metrics_tags,
)
from .rendering import StreamingRenderer
from .streaming import ARTalkStreamer, CausalSavgolSmoother


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FPS = 25
DEFAULT_RENDER_RES = 512
DEFAULT_RENDER_BATCH_SIZE = 8
AUDIO_OUT_PTIME = 0.020
AUDIO_OUT_SAMPLES_PER_FRAME = int(SAMPLE_RATE * AUDIO_OUT_PTIME)
AUDIO_SAMPLES_PER_VIDEO_FRAME = SAMPLE_RATE // FPS
DEFAULT_OUTPUT_AUDIO_PREBUFFER_SECONDS = 1.00
DEFAULT_OUTPUT_SEGMENT_SECONDS = 1.00


@dataclass
class QueuedAudioFrame:
    frame: av.AudioFrame
    accepted_at: float


@dataclass
class QueuedAudioSamples:
    samples: np.ndarray
    accepted_at: float


@dataclass
class PendingAudioChunk:
    samples: np.ndarray
    accepted_at: float

    def split(self, n_samples: int) -> tuple["PendingAudioChunk", "PendingAudioChunk"]:
        head = PendingAudioChunk(
            samples=self.samples[:n_samples],
            accepted_at=self.accepted_at,
        )
        tail = PendingAudioChunk(
            samples=self.samples[n_samples:],
            accepted_at=self.accepted_at + n_samples / SAMPLE_RATE,
        )
        return head, tail


@dataclass
class ConsumedAudioSlice:
    samples: np.ndarray
    first_accepted_at: float | None
    midpoint_accepted_at: float | None
    frame_midpoint_accepted_at: list[float]


@dataclass
class RenderedVideoFrame:
    image: np.ndarray
    motion_produced_at: float
    render_started_at: float
    render_finished_at: float
    audio_midpoint_accepted_at: float | None = None
    published_at: float | None = None
    frame_index: int | None = None


def with_pipeline_metrics(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.metrics_context():
            return method(self, *args, **kwargs)

    return wrapper


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
        output_audio_prebuffer_seconds=DEFAULT_OUTPUT_AUDIO_PREBUFFER_SECONDS,
        output_segment_seconds=DEFAULT_OUTPUT_SEGMENT_SECONDS,
        renderer_stage_sync=True,
        profile_trace_dir=None,
        profile_skip_chunks=1,
        profile_max_chunks=2,
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
            stage_sync=renderer_stage_sync,
        )
        self._renderer_stage_sync = bool(renderer_stage_sync)
        # PyTorch Profiler capture is opt-in and chunk-scoped: profiling every
        # 20 ms audio item would pay CUPTI start/stop costs continuously, so
        # the worker only profiles items expected to cross the model's chunk
        # boundary (the calls that run inference + rendering).
        self._profile_run_dir = None
        self._profile_skip_chunks = max(0, int(profile_skip_chunks))
        self._profile_max_chunks = max(0, int(profile_max_chunks))
        self._profile_chunks_seen = 0
        self._profile_chunks_captured = 0
        self._profiler_results: list[dict] = []
        self._profiler_results_lock = threading.Lock()
        self._profiler_last_error: str | None = None
        if profile_trace_dir:
            run_name = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"
            self._profile_run_dir = os.path.join(str(profile_trace_dir), run_name)
            os.makedirs(self._profile_run_dir, exist_ok=True)
            logger.warning(
                "[ARTalkPipeline] profiler traces will be written to %s "
                "(skip=%d, max=%d)",
                self._profile_run_dir,
                self._profile_skip_chunks,
                self._profile_max_chunks,
            )
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )
        # Video frames are indexed by ARTalk's 25 fps motion clock, not by
        # WebRTC callback count. The audio source callback advances the media
        # clock using the number of real delayed samples it has emitted; the
        # video source callback then serves the newest frame whose index is not
        # ahead of that audio clock. This prevents video from racing ahead when
        # rendering completes before enough delayed audio has actually played.
        self._video_queue: deque[tuple[int, RenderedVideoFrame]] = deque()
        self._video_queue_lock = threading.Lock()
        self._video_queue_max = 200
        self._next_video_frame_index = 0
        self._last_video_frame_index_served = -1
        self._audio_in_queue: queue.Queue = queue.Queue()
        # Output audio buffer: int16 mono samples at SAMPLE_RATE,
        # produced by the worker as it consumes input audio into the
        # streamer, drained by audio_source_callback at real-time
        # pacing so video and audio share the same ~4 s latency.
        self._audio_out_buffer = np.zeros(0, dtype=np.int16)
        self._audio_out_lock = threading.Lock()
        # Counts only real delayed audio samples emitted to the browser. Silence
        # used for startup, prebuffering, or underrun padding intentionally does
        # not advance this counter, so the video clock freezes instead of
        # drifting through silent gaps.
        self._synced_audio_samples_served = 0
        self._playback_started = False
        # Worker-only staging for input audio that the streamer has
        # accumulated but not yet "consumed" (i.e. produced motion
        # for). When motion fires we move the matching prefix into
        # _audio_out_buffer.
        self._pending_audio_for_output: list[PendingAudioChunk] = []
        self._stop_event = threading.Event()
        self._worker_busy = False
        self._dbg_calls = 0
        self._last_motion_streamer_elapsed_s = 0.0
        self._render_res = int(render_res)
        self._render_batch_size = max(1, int(render_batch_size))
        self._output_audio_prebuffer_seconds = max(
            0.0,
            float(output_audio_prebuffer_seconds),
        )
        self._output_audio_prebuffer_samples = int(
            SAMPLE_RATE * self._output_audio_prebuffer_seconds
        )
        self._output_segment_seconds = max(1 / FPS, float(output_segment_seconds))
        self._output_segment_min_frames = max(
            1,
            int(round(FPS * self._output_segment_seconds)),
        )
        self._metrics = PipelineMetrics()
        with self.metrics_context() as metrics:
            metrics.set("streamer_chunk_samples", self._streamer.patch_audio_length)
            metrics.set(
                "streamer_chunk_floor_s",
                self._streamer.patch_audio_length / SAMPLE_RATE,
            )
            metrics.set("streamer_frames_per_chunk", self._streamer.frames_per_chunk)
            metrics.set("fps", FPS)
            metrics.set("audio_out_samples_per_frame", AUDIO_OUT_SAMPLES_PER_FRAME)
            metrics.set(
                "audio_samples_per_video_frame",
                AUDIO_SAMPLES_PER_VIDEO_FRAME,
            )
            metrics.set(
                "output_audio_prebuffer_seconds",
                self._output_audio_prebuffer_seconds,
            )
            metrics.set(
                "output_audio_prebuffer_samples",
                self._output_audio_prebuffer_samples,
            )
            metrics.set("output_segment_seconds", self._output_segment_seconds)
            metrics.set("output_segment_min_frames", self._output_segment_min_frames)
            metrics.set("render_res", self._render_res)
            metrics.set("render_batch_size", self._render_batch_size)
            metrics.set("renderer_stage_sync", 1 if self._renderer_stage_sync else 0)
            metrics.set("profiler_enabled", 1 if self._profile_run_dir else 0)
        self._initial_placeholder = np.zeros(
            (self._render_res, self._render_res, 3),
            dtype=np.uint8,
        )
        self._placeholder = self._initial_placeholder
        with self.metrics_context():
            if self._renderer.mode == "gagavatar":
                self._warm_up_renderer()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ARTalkPipelineWorker",
            daemon=True,
        )
        self._worker_thread.start()

    @contextmanager
    def metrics_context(self):
        with pipeline_metrics_context(self._metrics) as metrics:
            yield metrics

    def metrics_snapshot(self) -> dict:
        return self._metrics.snapshot()

    @with_pipeline_metrics
    def push_audio_frame(self, frame: av.AudioFrame):
        """Fast: enqueue the raw frame, return. The worker handles
        resampling to 16 kHz mono and the heavy inference / render."""
        metrics = current_pipeline_metrics()
        accepted_at = time.perf_counter()
        metrics.set_once("first_audio_push_s", accepted_at)
        metrics.inc("audio_frames_pushed")
        self._record_frame_timestamp("input_audio", frame)
        self._audio_in_queue.put(QueuedAudioFrame(frame=frame, accepted_at=accepted_at))
        depth = self._audio_in_queue.qsize()
        metrics.set("audio_in_queue_depth", depth)

    @with_pipeline_metrics
    def push_audio_samples(self, samples_int16: np.ndarray) -> None:
        """Queue already-resampled 16 kHz mono int16 samples.

        This is useful for app-level glue code that synthesizes filler audio
        between upstream audio chunks. The samples still flow through the
        pipeline worker so ARTalkStreamer, smoothing, and rendering state remain
        single-threaded.
        """
        samples = np.asarray(samples_int16, dtype=np.int16)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        if samples.size == 0:
            return
        metrics = current_pipeline_metrics()
        accepted_at = time.perf_counter()
        metrics.set_once("first_audio_push_s", accepted_at)
        metrics.inc("audio_sample_chunks_pushed")
        metrics.inc("audio_samples_pushed", samples.size)
        self._audio_in_queue.put(
            QueuedAudioSamples(samples=samples.copy(), accepted_at=accepted_at)
        )
        depth = self._audio_in_queue.qsize()
        metrics.set("audio_in_queue_depth", depth)

    def push_silence(self, duration_s: float) -> None:
        n_samples = max(0, int(round(duration_s * SAMPLE_RATE)))
        if n_samples:
            self.push_audio_samples(np.zeros(n_samples, dtype=np.int16))

    def output_buffer_snapshot(self) -> dict[str, int]:
        with self._audio_out_lock:
            audio_out_buffer_samples = self._audio_out_buffer.size
        return {
            "audio_in_queue_depth": self._audio_in_queue.qsize(),
            "audio_out_buffer_samples": audio_out_buffer_samples,
            "video_queue_depth": self._video_queue_depth(),
            "worker_busy": int(self._worker_busy),
        }

    @with_pipeline_metrics
    def video_source_callback(
        self, pts: int, time_base: fractions.Fraction
    ) -> av.VideoFrame:
        """Synchronous frame producer for streamlit-webrtc's
        ``create_video_source_track``. Drains one frame from
        ``video_queue`` if available, falls back to a black placeholder
        otherwise — the call must return promptly so the outbound
        track keeps firing at its configured fps.
        """
        metrics = current_pipeline_metrics()
        metrics.set_once(
            "first_video_callback_s",
            time.perf_counter(),
        )
        metrics.inc("video_callbacks")
        self._record_callback_timestamp("video_source", pts, time_base)
        target_index = self._media_clock_frame_index()
        rendered_frame, frame_index = self._pop_video_frame_for_media_clock(target_index)
        metrics.set("last_video_target_frame_index", target_index)
        if rendered_frame is None:
            arr = self._placeholder
            metrics.inc("video_placeholder_frames")
            metrics.inc("video_no_ready_frame_callbacks")
        else:
            arr = rendered_frame.image
            self._placeholder = arr
            self._last_video_frame_index_served = frame_index
            metrics.inc("video_frames_served")
            metrics.set("last_video_frame_index_served", frame_index)
            self._record_served_video_lag(target_index, frame_index)
            self._record_served_frame_latency(rendered_frame)
        metrics.set("video_queue_depth", self._video_queue_depth())
        self._record_video_lead(target_index)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    @with_pipeline_metrics
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
        metrics = current_pipeline_metrics()
        metrics.set_once(
            "first_audio_callback_s",
            time.perf_counter(),
        )
        metrics.inc("audio_callbacks")
        self._record_callback_timestamp("audio_source", pts, time_base)
        n = AUDIO_OUT_SAMPLES_PER_FRAME
        playback_started_now = False
        with self._audio_out_lock:
            available = self._audio_out_buffer.size
            if (
                not self._playback_started
                and available >= self._output_audio_prebuffer_samples
            ):
                self._playback_started = True
                playback_started_now = True
            # Do not emit partial audio padded with zeros. That made chunk
            # boundaries audible as clicks/stutters. If the buffer cannot
            # satisfy a complete WebRTC audio frame, emit silence and keep the
            # real-audio media clock stationary until a complete frame is ready.
            if self._playback_started and available >= n:
                samples = self._audio_out_buffer[:n].copy()
                self._audio_out_buffer = self._audio_out_buffer[n:]
                underrun = 0
            else:
                samples = np.zeros(n, dtype=np.int16)
                underrun = n
            real_samples = n - underrun
            if real_samples:
                self._synced_audio_samples_served += real_samples
            synced_audio_samples = self._synced_audio_samples_served
            buffered = self._audio_out_buffer.size
        metrics.inc("audio_frames_served")
        if playback_started_now:
            metrics.inc("audio_playback_starts")
        if underrun:
            if not self._playback_started:
                metrics.inc("audio_preplayback_silence_frames")
                metrics.inc("audio_preplayback_silence_samples", underrun)
            elif available > 0:
                metrics.inc("audio_short_buffer_frames")
                metrics.inc("audio_short_buffer_samples", underrun)
            else:
                metrics.inc("audio_playback_underrun_frames")
                metrics.inc("audio_playback_underrun_samples", underrun)
        metrics.set("audio_playback_started", 1 if self._playback_started else 0)
        metrics.set("synced_audio_samples_served", synced_audio_samples)
        metrics.set(
            "synced_audio_frame_index",
            synced_audio_samples // AUDIO_SAMPLES_PER_VIDEO_FRAME,
        )
        metrics.set("audio_out_buffer_samples", buffered)
        metrics.set("audio_out_buffer_seconds", buffered / SAMPLE_RATE)
        metrics.observe_min(
            "min_audio_out_buffer_seconds",
            buffered / SAMPLE_RATE,
        )
        metrics.observe_max(
            "max_audio_out_buffer_seconds",
            buffered / SAMPLE_RATE,
        )
        # AudioFrame.from_ndarray for s16 mono expects shape (1, N).
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def _media_clock_frame_index(self) -> int:
        with self._audio_out_lock:
            samples = self._synced_audio_samples_served
        if samples <= 0:
            return -1
        return int(samples // AUDIO_SAMPLES_PER_VIDEO_FRAME)

    def _pop_video_frame_for_media_clock(
        self,
        target_index: int,
    ) -> tuple[RenderedVideoFrame | None, int | None]:
        # The browser asks for video at a steady cadence even when rendering is
        # bursty. If rendering fell behind and then catches up, multiple queued
        # frames can be older than the audio clock. Drop those stale frames and
        # serve the latest frame at or before the current audio-derived index.
        dropped_for_sync = 0
        with self._video_queue_lock:
            selected: tuple[int, RenderedVideoFrame] | None = None
            while self._video_queue and self._video_queue[0][0] <= target_index:
                selected = self._video_queue.popleft()
                if self._video_queue and self._video_queue[0][0] <= target_index:
                    dropped_for_sync += 1
            depth = len(self._video_queue)
        metrics = current_pipeline_metrics()
        if dropped_for_sync:
            metrics.inc("video_frames_dropped_for_sync", dropped_for_sync)
        metrics.set("video_queue_depth", depth)
        if selected is None:
            return None, None
        return selected[1], selected[0]

    def _record_served_frame_latency(self, frame: RenderedVideoFrame) -> None:
        if frame.audio_midpoint_accepted_at is None:
            return
        served_at = time.perf_counter()
        metrics = current_pipeline_metrics()
        served_latency_s = served_at - frame.audio_midpoint_accepted_at
        publish_to_serve_s = (
            served_at - frame.published_at
            if frame.published_at is not None
            else 0.0
        )
        render_to_serve_s = served_at - frame.render_finished_at
        metrics.set("last_frame_audio_to_video_served_latency_s", served_latency_s)
        metrics.set("last_frame_publish_to_serve_latency_s", publish_to_serve_s)
        metrics.set("last_frame_render_to_serve_latency_s", render_to_serve_s)
        metrics.observe_ms("frame_audio_to_video_served_latency", served_latency_s)
        metrics.observe_ms("frame_publish_to_serve_latency", publish_to_serve_s)
        metrics.observe_ms("frame_render_to_serve_latency", render_to_serve_s)
        metrics.observe_min("min_frame_audio_to_video_served_latency_s", served_latency_s)
        metrics.observe_max("max_frame_audio_to_video_served_latency_s", served_latency_s)

    def _record_served_video_lag(
        self,
        target_index: int,
        frame_index: int | None,
    ) -> None:
        if frame_index is None or target_index < 0:
            return
        metrics = current_pipeline_metrics()
        lag_frames = target_index - frame_index
        metrics.set("last_video_served_lag_frames", lag_frames)
        metrics.set("last_video_served_lag_seconds", lag_frames / FPS)
        metrics.observe_max("max_video_served_lag_frames", lag_frames)

    def _record_video_lead(self, target_index: int) -> None:
        with self._video_queue_lock:
            depth = len(self._video_queue)
            first_index = self._video_queue[0][0] if self._video_queue else None
            last_index = self._video_queue[-1][0] if self._video_queue else None
            last_enqueued = self._next_video_frame_index - 1
        metrics = current_pipeline_metrics()
        metrics.set("video_queue_depth", depth)
        if first_index is not None:
            metrics.set("video_queue_first_frame_index", first_index)
        if last_index is not None:
            metrics.set("video_queue_last_frame_index", last_index)
            metrics.set("video_queue_span_frames", last_index - first_index + 1)
            metrics.set(
                "video_queue_span_seconds",
                (last_index - first_index + 1) / FPS,
            )
        if target_index < 0 or last_enqueued < 0:
            return
        lead_frames = last_enqueued - target_index
        metrics.set("video_lead_frames", lead_frames)
        metrics.set("video_lead_seconds", lead_frames / FPS)
        metrics.observe_min("min_video_lead_frames", lead_frames)
        metrics.observe_max("max_video_lead_frames", lead_frames)

    def _record_frame_timestamp(self, prefix: str, frame) -> None:
        if frame.pts is None:
            return
        metrics = current_pipeline_metrics()
        metrics.set(f"last_{prefix}_pts", frame.pts)
        if frame.time_base is not None:
            time_base = float(frame.time_base)
            metrics.set(f"last_{prefix}_time_base", time_base)
            metrics.set(f"last_{prefix}_time_s", frame.pts * time_base)

    def _record_callback_timestamp(
        self,
        prefix: str,
        pts: int | None,
        time_base: fractions.Fraction,
    ) -> None:
        if pts is None:
            return
        metrics = current_pipeline_metrics()
        metrics.set(f"last_{prefix}_pts", pts)
        time_base_s = float(time_base)
        metrics.set(f"last_{prefix}_time_base", time_base_s)
        metrics.set(f"last_{prefix}_time_s", pts * time_base_s)

    @with_pipeline_metrics
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
            self._synced_audio_samples_served = 0
            self._playback_started = False
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
        with self._video_queue_lock:
            self._video_queue.clear()
            self._next_video_frame_index = 0
            self._last_video_frame_index_served = -1
        metrics = current_pipeline_metrics()
        metrics.set("video_queue_depth", 0)

    @staticmethod
    def _drain_queue(q: queue.Queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    @with_pipeline_metrics
    def flush_output(self):
        """Drop queued-but-unplayed output and skip the media clock past it.

        For barge-in: when the user interrupts, already-published response
        audio/video should stop playing immediately. The flushed audio is
        credited to the served-samples clock so that frame indices published
        afterwards stay aligned with it — without the credit the clock could
        never reach them and video would freeze permanently. Only
        callback-side state is touched, so this is safe to call from any
        thread; audio already inside the worker (streamer buffer plus the
        chunk currently rendering) still publishes, bounded by one model
        chunk of stale content.
        """
        metrics = current_pipeline_metrics()
        self._drain_queue(self._audio_in_queue)
        with self._audio_out_lock:
            flushed_samples = int(self._audio_out_buffer.size)
            self._audio_out_buffer = np.zeros(0, dtype=np.int16)
            self._synced_audio_samples_served += flushed_samples
            synced_audio_samples = self._synced_audio_samples_served
        with self._video_queue_lock:
            flushed_frames = len(self._video_queue)
            self._video_queue.clear()
        metrics.inc("output_flushes")
        metrics.inc("flushed_audio_samples", flushed_samples)
        metrics.inc("flushed_video_frames", flushed_frames)
        metrics.set("audio_out_buffer_samples", 0)
        metrics.set("audio_out_buffer_seconds", 0.0)
        metrics.set("video_queue_depth", 0)
        metrics.set("synced_audio_samples_served", synced_audio_samples)
        metrics.set(
            "synced_audio_frame_index",
            synced_audio_samples // AUDIO_SAMPLES_PER_VIDEO_FRAME,
        )

    def _pop_pending_audio_for_output(self, n_samples: int) -> list[PendingAudioChunk]:
        # ARTalk consumes audio in large model chunks before it can produce
        # motion. Keep the raw input audio staged until the corresponding
        # smoothed motion frames are known, then pop exactly the samples that
        # map to those rendered frames. When a chunk is split, the tail timestamp
        # advances by media duration so latency can be attributed to the
        # relevant video frames rather than the parent chunk's first sample.
        remaining = max(0, n_samples)
        emitted: list[PendingAudioChunk] = []
        while remaining and self._pending_audio_for_output:
            chunk = self._pending_audio_for_output.pop(0)
            if chunk.samples.size <= remaining:
                emitted.append(chunk)
                remaining -= chunk.samples.size
                continue
            head, tail = chunk.split(remaining)
            emitted.append(head)
            self._pending_audio_for_output.insert(0, tail)
            remaining = 0
        pending_samples = sum(chunk.samples.size for chunk in self._pending_audio_for_output)
        metrics = current_pipeline_metrics()
        metrics.set("pending_audio_for_output_samples", pending_samples)
        return emitted

    @staticmethod
    def _chunk_timestamp_at(chunk: PendingAudioChunk, sample_offset: int) -> float:
        return chunk.accepted_at + sample_offset / SAMPLE_RATE

    @classmethod
    def _timestamp_at_consumed_sample(
        cls,
        consumed_chunks: list[PendingAudioChunk],
        sample_index: int,
    ) -> float | None:
        cursor = 0
        for chunk in consumed_chunks:
            next_cursor = cursor + chunk.samples.size
            if sample_index < next_cursor:
                return cls._chunk_timestamp_at(chunk, sample_index - cursor)
            cursor = next_cursor
        return None

    @classmethod
    def _consume_audio_chunks(
        cls,
        chunks: list[PendingAudioChunk],
        n_samples: int,
    ) -> ConsumedAudioSlice:
        remaining = max(0, n_samples)
        consumed: list[np.ndarray] = []
        consumed_chunks: list[PendingAudioChunk] = []
        accepted_at: float | None = None
        while remaining and chunks:
            chunk = chunks.pop(0)
            if accepted_at is None:
                accepted_at = chunk.accepted_at
            if chunk.samples.size <= remaining:
                consumed.append(chunk.samples)
                consumed_chunks.append(chunk)
                remaining -= chunk.samples.size
                continue
            head, tail = chunk.split(remaining)
            consumed.append(head.samples)
            consumed_chunks.append(head)
            chunks.insert(0, tail)
            remaining = 0
        if not consumed:
            return ConsumedAudioSlice(
                samples=np.zeros(0, dtype=np.int16),
                first_accepted_at=accepted_at,
                midpoint_accepted_at=accepted_at,
                frame_midpoint_accepted_at=[],
            )
        total_samples = sum(chunk.samples.size for chunk in consumed_chunks)
        midpoint_sample = total_samples // 2
        midpoint_accepted_at = cls._timestamp_at_consumed_sample(
            consumed_chunks,
            midpoint_sample,
        )
        frame_midpoint_accepted_at = []
        for start in range(0, total_samples, AUDIO_SAMPLES_PER_VIDEO_FRAME):
            midpoint = min(
                start + AUDIO_SAMPLES_PER_VIDEO_FRAME // 2,
                total_samples - 1,
            )
            timestamp = cls._timestamp_at_consumed_sample(consumed_chunks, midpoint)
            if timestamp is not None:
                frame_midpoint_accepted_at.append(timestamp)
        return ConsumedAudioSlice(
            samples=np.concatenate(consumed),
            first_accepted_at=accepted_at,
            midpoint_accepted_at=midpoint_accepted_at,
            frame_midpoint_accepted_at=frame_midpoint_accepted_at,
        )

    def _publish_output_segment(
        self,
        frames: list[RenderedVideoFrame],
        audio_slice: ConsumedAudioSlice,
        motion_produced_at: float,
        segment_render_started_at: float | None,
        segment_media_offset_s: float = 0.0,
    ) -> None:
        audio = audio_slice.samples
        if not frames and audio.size == 0:
            return
        published_at = time.perf_counter()
        # Keep callback locks short. Video is queued before audio, but video
        # display is driven by the real-audio clock, so those frames remain
        # invisible until the matching audio buffer is published and drained by
        # audio_source_callback. Holding both locks together made WebRTC
        # callbacks occasionally wait through segment publication.
        metrics = current_pipeline_metrics()
        dropped_video_frames = 0
        last_frame_index = None
        frame_timestamps = audio_slice.frame_midpoint_accepted_at[: len(frames)]
        for frame, accepted_at in zip(frames, frame_timestamps):
            frame.audio_midpoint_accepted_at = accepted_at
            frame.published_at = published_at
        publish_video_t0 = time.perf_counter()
        with self._video_queue_lock:
            for frame in frames:
                frame_index = self._next_video_frame_index
                self._next_video_frame_index += 1
                frame.frame_index = frame_index
                while len(self._video_queue) >= self._video_queue_max:
                    self._video_queue.popleft()
                    dropped_video_frames += 1
                self._video_queue.append((frame_index, frame))
                last_frame_index = frame_index
            video_depth = len(self._video_queue)
            front_frame_index = self._video_queue[0][0] if self._video_queue else None
        metrics.observe_ms(
            "post_model_publish_video_queue",
            time.perf_counter() - publish_video_t0,
        )
        if dropped_video_frames and front_frame_index is not None:
            # The video queue is bounded but the audio buffer is not; without
            # this, audio paired with the evicted frames still plays and the
            # video freezes until the clock reaches the surviving frames.
            # Fast-forward playback to the surviving front frame instead.
            with self._audio_out_lock:
                skip = (
                    front_frame_index * AUDIO_SAMPLES_PER_VIDEO_FRAME
                    - self._synced_audio_samples_served
                )
                skip = min(max(skip, 0), int(self._audio_out_buffer.size))
                if skip:
                    self._audio_out_buffer = self._audio_out_buffer[skip:]
                    self._synced_audio_samples_served += skip
                synced_audio_samples = self._synced_audio_samples_served
            if skip:
                metrics.inc("audio_samples_skipped_for_dropped_video", skip)
                metrics.set("synced_audio_samples_served", synced_audio_samples)
                metrics.set(
                    "synced_audio_frame_index",
                    synced_audio_samples // AUDIO_SAMPLES_PER_VIDEO_FRAME,
                )
        publish_audio_t0 = time.perf_counter()
        with self._audio_out_lock:
            if audio.size:
                self._audio_out_buffer = np.concatenate(
                    [self._audio_out_buffer, audio]
                )
            audio_buffered = self._audio_out_buffer.size
        metrics.observe_ms(
            "post_model_publish_audio_buffer",
            time.perf_counter() - publish_audio_t0,
        )
        publish_metrics_t0 = time.perf_counter()
        if dropped_video_frames:
            metrics.inc("video_frames_dropped", dropped_video_frames)
        if last_frame_index is not None:
            metrics.set("last_video_frame_index_enqueued", last_frame_index)
        metrics.set("video_queue_depth", video_depth)
        metrics.set("audio_out_buffer_samples", audio_buffered)
        metrics.set("audio_out_buffer_seconds", audio_buffered / SAMPLE_RATE)
        metrics.observe_min(
            "min_audio_out_buffer_seconds",
            audio_buffered / SAMPLE_RATE,
        )
        metrics.observe_max(
            "max_audio_out_buffer_seconds",
            audio_buffered / SAMPLE_RATE,
        )
        metrics.inc("output_segments_published")
        metrics.inc("output_segment_frames", len(frames))
        metrics.inc("output_segment_audio_samples", audio.size)
        metrics.set("last_output_segment_frames", len(frames))
        metrics.set("last_output_segment_audio_samples", audio.size)
        if frames and audio_slice.first_accepted_at is not None:
            pre_model_compute_s = self._last_motion_streamer_elapsed_s
            frames_with_audio = [
                frame for frame in frames if frame.audio_midpoint_accepted_at is not None
            ]
            frame_latencies_s = [
                published_at - frame.audio_midpoint_accepted_at
                for frame in frames_with_audio
            ]
            frame_pre_model_s = [
                frame.motion_produced_at - frame.audio_midpoint_accepted_at
                for frame in frames_with_audio
            ]
            frame_render_s = [
                frame.render_finished_at - frame.render_started_at
                for frame in frames_with_audio
            ]
            frame_render_to_publish_s = [
                published_at - frame.render_finished_at for frame in frames_with_audio
            ]
            frame_pre_model_wait_s = [
                max(0.0, latency_s - pre_model_compute_s)
                for latency_s in frame_pre_model_s
            ]
            for latency_s in frame_latencies_s:
                metrics.observe_ms("frame_audio_to_video_latency", latency_s)
                metrics.observe_min("min_frame_audio_to_video_latency_s", latency_s)
                metrics.observe_max("max_frame_audio_to_video_latency_s", latency_s)
            for latency_s in frame_pre_model_s:
                metrics.observe_ms("frame_pre_model_latency", latency_s)
            for latency_s in frame_render_s:
                metrics.observe_ms("frame_render_latency", latency_s)
            for latency_s in frame_render_to_publish_s:
                metrics.observe_ms("frame_render_to_publish_latency", latency_s)
            if frame_latencies_s:
                mid_index = len(frame_latencies_s) // 2
                frame_first_latency_s = frame_latencies_s[0]
                frame_midpoint_latency_s = frame_latencies_s[mid_index]
                frame_last_latency_s = frame_latencies_s[-1]
                frame_pre_model_first_s = frame_pre_model_s[0]
                frame_pre_model_midpoint_s = frame_pre_model_s[mid_index]
                frame_pre_model_last_s = frame_pre_model_s[-1]
                frame_pre_model_wait_first_s = frame_pre_model_wait_s[0]
                frame_pre_model_wait_midpoint_s = frame_pre_model_wait_s[mid_index]
                frame_pre_model_wait_last_s = frame_pre_model_wait_s[-1]
                frame_render_midpoint_s = frame_render_s[mid_index]
                frame_render_to_publish_midpoint_s = frame_render_to_publish_s[mid_index]
                metrics.set(
                    "last_frame_audio_to_video_first_latency_s",
                    frame_first_latency_s,
                )
                metrics.set(
                    "last_frame_audio_to_video_midpoint_latency_s",
                    frame_midpoint_latency_s,
                )
                metrics.set(
                    "last_frame_audio_to_video_last_latency_s",
                    frame_last_latency_s,
                )
                metrics.set("last_frame_pre_model_first_latency_s", frame_pre_model_first_s)
                metrics.set(
                    "last_frame_pre_model_midpoint_latency_s",
                    frame_pre_model_midpoint_s,
                )
                metrics.set("last_frame_pre_model_last_latency_s", frame_pre_model_last_s)
                metrics.set(
                    "last_frame_pre_model_wait_first_s",
                    frame_pre_model_wait_first_s,
                )
                metrics.set(
                    "last_frame_pre_model_wait_midpoint_s",
                    frame_pre_model_wait_midpoint_s,
                )
                metrics.set(
                    "last_frame_pre_model_wait_last_s",
                    frame_pre_model_wait_last_s,
                )
                metrics.set("last_frame_render_midpoint_latency_s", frame_render_midpoint_s)
                metrics.set(
                    "last_frame_render_to_publish_midpoint_latency_s",
                    frame_render_to_publish_midpoint_s,
                )
                metrics.observe_ms(
                    "frame_audio_to_video_first_latency",
                    frame_first_latency_s,
                )
                metrics.observe_ms(
                    "frame_audio_to_video_midpoint_latency",
                    frame_midpoint_latency_s,
                )
                metrics.observe_ms(
                    "frame_audio_to_video_last_latency",
                    frame_last_latency_s,
                )
                metrics.observe_ms(
                    "frame_pre_model_first_latency",
                    frame_pre_model_first_s,
                )
                metrics.observe_ms(
                    "frame_pre_model_midpoint_latency",
                    frame_pre_model_midpoint_s,
                )
                metrics.observe_ms(
                    "frame_pre_model_last_latency",
                    frame_pre_model_last_s,
                )
                metrics.observe_ms(
                    "frame_pre_model_wait_first",
                    frame_pre_model_wait_first_s,
                )
                metrics.observe_ms(
                    "frame_pre_model_wait_midpoint",
                    frame_pre_model_wait_midpoint_s,
                )
                metrics.observe_ms(
                    "frame_pre_model_wait_last",
                    frame_pre_model_wait_last_s,
                )
                metrics.observe_ms(
                    "frame_render_midpoint_latency",
                    frame_render_midpoint_s,
                )
                metrics.observe_ms(
                    "frame_render_to_publish_midpoint_latency",
                    frame_render_to_publish_midpoint_s,
                )
            audio_to_video_latency_s = published_at - audio_slice.first_accepted_at
            midpoint_latency_s = (
                published_at - audio_slice.midpoint_accepted_at
                if audio_slice.midpoint_accepted_at is not None
                else audio_to_video_latency_s
            )
            latency_anchor_at = (
                audio_slice.midpoint_accepted_at
                if audio_slice.midpoint_accepted_at is not None
                else audio_slice.first_accepted_at
            )
            pre_model_s = motion_produced_at - latency_anchor_at
            post_model_s = published_at - motion_produced_at
            render_s = (
                published_at - segment_render_started_at
                if segment_render_started_at is not None
                else 0.0
            )
            pre_render_wait_s = post_model_s - render_s
            post_model_excess_s = max(0.0, post_model_s - segment_media_offset_s)
            pre_model_wait_s = max(0.0, pre_model_s - pre_model_compute_s)
            post_model_publish_overhead_s = max(
                0.0,
                post_model_s - pre_render_wait_s - render_s,
            )
            metrics.set(
                "last_audio_to_video_latency_s",
                audio_to_video_latency_s,
            )
            metrics.set(
                "last_midpoint_audio_to_video_latency_s",
                midpoint_latency_s,
            )
            metrics.set("last_pre_model_latency_s", pre_model_s)
            metrics.set("last_post_model_latency_s", post_model_s)
            metrics.set("last_post_model_excess_latency_s", post_model_excess_s)
            metrics.set("last_pre_render_wait_latency_s", pre_render_wait_s)
            metrics.set("last_segment_render_latency_s", render_s)
            metrics.set("last_pre_model_chunk_wait_s", pre_model_wait_s)
            metrics.set("last_pre_model_streamer_compute_s", pre_model_compute_s)
            metrics.set("last_post_model_render_window_s", render_s)
            metrics.set(
                "last_post_model_publish_overhead_s",
                post_model_publish_overhead_s,
            )
            metrics.observe_ms(
                "audio_to_video_latency",
                audio_to_video_latency_s,
            )
            metrics.observe_ms(
                "audio_midpoint_to_video_latency",
                midpoint_latency_s,
            )
            metrics.observe_ms("pre_model_latency", pre_model_s)
            metrics.observe_ms("post_model_latency", post_model_s)
            metrics.observe_ms("post_model_excess_latency", post_model_excess_s)
            metrics.observe_ms("pre_render_wait_latency", pre_render_wait_s)
            metrics.observe_ms("segment_render_latency", render_s)
            metrics.observe_ms("pre_model_chunk_wait", pre_model_wait_s)
            metrics.observe_ms("pre_model_streamer_compute", pre_model_compute_s)
            metrics.observe_ms("post_model_render_window", render_s)
            metrics.observe_ms(
                "post_model_publish_overhead",
                post_model_publish_overhead_s,
            )
            metrics.observe_min(
                "min_audio_to_video_latency_s",
                audio_to_video_latency_s,
            )
            metrics.observe_max(
                "max_audio_to_video_latency_s",
                audio_to_video_latency_s,
            )
        metrics.observe_ms(
            "post_model_publish_metrics_update",
            time.perf_counter() - publish_metrics_t0,
        )
        self._record_video_lead(self._media_clock_frame_index())

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _warm_up_renderer(self):
        metrics = current_pipeline_metrics()
        motion = torch.zeros(
            self._streamer.motion_dim,
            dtype=self._streamer.dtype,
            device=self._streamer.device,
        )
        with observe_pipeline_duration("renderer_warmup"):
            rgb, timings = self._renderer.render_frame_profile(motion)
            convert_t0 = time.perf_counter()
            (
                (rgb * 255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .numpy()
            )
            metrics.observe_ms(
                "warmup_rgb_tensor_to_numpy",
                time.perf_counter() - convert_t0,
            )
        for key, elapsed_s in timings.items():
            metrics.observe_ms(f"warmup_{key}", elapsed_s)

    def _worker_loop(self):
        with self.metrics_context():
            while not self._stop_event.is_set():
                try:
                    item = self._audio_in_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                metrics = current_pipeline_metrics()
                metrics.set(
                    "audio_in_queue_depth",
                    self._audio_in_queue.qsize(),
                )
                self._worker_busy = True
                metrics.set("worker_busy", 1)
                try:
                    if self._should_profile_item(item):
                        self._process_worker_item_profiled(item)
                    else:
                        self._process_worker_item(item)
                except Exception:
                    logger.exception("[ARTalkPipeline] worker error")
                finally:
                    self._worker_busy = False
                    metrics.set("worker_busy", 0)

    def _process_worker_item(self, item):
        if isinstance(item, QueuedAudioSamples):
            self._process_sample_chunk(
                item.samples,
                item.accepted_at,
            )
        elif isinstance(item, QueuedAudioFrame):
            self._process_input_frame(
                item.frame,
                item.accepted_at,
            )
        elif isinstance(item, np.ndarray):
            self._process_sample_chunk(
                item,
                time.perf_counter(),
            )
        else:
            self._process_input_frame(
                item,
                time.perf_counter(),
            )

    def _estimate_item_samples_16k(self, item) -> int:
        """Estimate how many 16 kHz samples ``item`` contributes post-resample."""
        if isinstance(item, QueuedAudioSamples):
            return int(item.samples.size)
        if isinstance(item, np.ndarray):
            return int(item.size)
        frame = item.frame if isinstance(item, QueuedAudioFrame) else item
        n_samples = getattr(frame, "samples", 0) or 0
        rate = getattr(frame, "sample_rate", 0) or SAMPLE_RATE
        return int(n_samples * SAMPLE_RATE / rate)

    def _should_profile_item(self, item) -> bool:
        if self._profile_run_dir is None:
            return False
        if self._profile_chunks_captured >= self._profile_max_chunks:
            return False
        buffered = self._streamer._audio_buffer.shape[0]
        incoming = self._estimate_item_samples_16k(item)
        if buffered + incoming < self._streamer.patch_audio_length:
            return False
        self._profile_chunks_seen += 1
        return self._profile_chunks_seen > self._profile_skip_chunks

    def _process_worker_item_profiled(self, item):
        metrics = current_pipeline_metrics()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(self._device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Profiler failures must never take the audio item down with them:
        # process the item unprofiled when the profiler cannot start, and
        # record the error so the UI can surface it instead of appearing to
        # wait forever.
        try:
            prof = torch.profiler.profile(activities=activities)
            prof.__enter__()
        except Exception as exc:
            logger.exception("[ARTalkPipeline] profiler start failed")
            self._set_profiler_error(f"{type(exc).__name__}: {exc}")
            self._process_worker_item(item)
            return
        stop_failed = False
        try:
            self._process_worker_item(item)
        finally:
            try:
                prof.__exit__(None, None, None)
            except Exception as exc:
                logger.exception("[ARTalkPipeline] profiler stop failed")
                self._set_profiler_error(f"{type(exc).__name__}: {exc}")
                stop_failed = True
        if stop_failed:
            return
        self._profile_chunks_captured += 1
        # Export + summary can take seconds for a 4-second chunk trace; doing
        # that on the worker thread stalls audio/video output right after the
        # capture, so hand the finished profile off to a background thread.
        threading.Thread(
            target=self._export_profiler_capture,
            args=(prof, self._profile_chunks_seen, self._profile_chunks_captured),
            name="ARTalkProfilerExport",
            daemon=True,
        ).start()

    def _export_profiler_capture(self, prof, chunk_index, capture_index):
        with self.metrics_context():
            metrics = current_pipeline_metrics()
            trace_path = os.path.join(
                self._profile_run_dir,
                f"chunk-{chunk_index:03d}.json",
            )
            export_t0 = time.perf_counter()
            try:
                prof.export_chrome_trace(trace_path)
            except Exception as exc:
                logger.exception("[ARTalkPipeline] profiler trace export failed")
                self._set_profiler_error(f"{type(exc).__name__}: {exc}")
                return
            metrics.observe_ms(
                "profiler_trace_export", time.perf_counter() - export_t0
            )
            metrics.inc("profiler_traces_captured")
            summary_t0 = time.perf_counter()
            sort_by = (
                "self_cuda_time_total"
                if torch.device(self._device).type == "cuda"
                else "self_cpu_time_total"
            )
            try:
                summary_table = prof.key_averages().table(
                    sort_by=sort_by, row_limit=30
                )
            except Exception:
                logger.exception("[ARTalkPipeline] profiler summary build failed")
                summary_table = ""
            metrics.observe_ms(
                "profiler_summary_build", time.perf_counter() - summary_t0
            )
            if summary_table:
                # Persist next to the trace so UIs can show summaries from
                # earlier pipeline instances (the in-memory results below die
                # with this pipeline on session restarts).
                summary_path = os.path.splitext(trace_path)[0] + "-summary.txt"
                try:
                    with open(summary_path, "w") as f:
                        f.write(summary_table)
                except OSError:
                    logger.exception(
                        "[ARTalkPipeline] profiler summary write failed"
                    )
            with self._profiler_results_lock:
                self._profiler_results.append(
                    {
                        "chunk_index": chunk_index,
                        "trace_path": trace_path,
                        "summary_table": summary_table,
                        "captured_at_s": time.time(),
                    }
                )
            logger.warning(
                "[ARTalkPipeline] profiler trace exported (%d/%d): %s",
                capture_index,
                self._profile_max_chunks,
                trace_path,
            )

    def _set_profiler_error(self, message: str) -> None:
        with self._profiler_results_lock:
            self._profiler_last_error = message

    @property
    def profiler_run_dir(self) -> str | None:
        return self._profile_run_dir

    @property
    def profiler_max_chunks(self) -> int:
        return self._profile_max_chunks

    def profiler_status(self) -> dict:
        """Snapshot of profiler configuration and capture progress."""
        with self._profiler_results_lock:
            last_error = self._profiler_last_error
        return {
            "run_dir": self._profile_run_dir,
            "skip_chunks": self._profile_skip_chunks,
            "max_chunks": self._profile_max_chunks,
            "chunks_seen": self._profile_chunks_seen,
            "chunks_captured": self._profile_chunks_captured,
            "last_error": last_error,
        }

    def profiler_results(self) -> list[dict]:
        """Snapshot of captured profiler summaries (oldest first)."""
        with self._profiler_results_lock:
            return list(self._profiler_results)

    def _process_input_frame(
        self,
        frame: av.AudioFrame,
        accepted_at: float,
    ):
        # Resample to 16 kHz mono int16 once; the same chunks are both
        # staged for outbound audio (delayed-emit, paired with rendered
        # video below) and fed to the streamer for motion inference.
        metrics = current_pipeline_metrics()
        with observe_pipeline_duration("resample"):
            with torch.profiler.record_function("artalk.resample"):
                resampled = self._resampler.resample(frame)
        if not isinstance(resampled, list):
            resampled = [resampled]
        for rf in resampled:
            arr = rf.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            if arr.size == 0:
                continue
            samples = arr.astype(np.int16, copy=True)
            metrics.inc("audio_chunks_resampled")
            metrics.inc("audio_samples_resampled", samples.size)
            self._process_sample_chunk(samples, accepted_at)

    def _process_sample_chunk(
        self,
        samples_int16: np.ndarray,
        accepted_at: float,
    ) -> None:
        self._pending_audio_for_output.append(
            PendingAudioChunk(
                samples=samples_int16,
                accepted_at=accepted_at,
            )
        )
        self._process_audio_chunk(samples_int16)

    def _process_audio_chunk(self, samples_int16: np.ndarray):
        metrics = current_pipeline_metrics()
        metrics.set_once("first_audio_process_s", time.perf_counter())
        samples_t = torch.from_numpy(
            samples_int16.astype(np.float32) / 32768.0
        ).to(self._device)
        buf_before = self._streamer._audio_buffer.shape[0]
        streamer_t0 = time.perf_counter()
        with torch.profiler.record_function("artalk.streamer_feed"):
            motion = self._streamer.feed(samples_t)
        streamer_elapsed_s = time.perf_counter() - streamer_t0
        metrics.observe_ms("artalk_streamer_feed", streamer_elapsed_s)
        buf_after = self._streamer._audio_buffer.shape[0]
        metrics.inc("streamer_feed_calls")
        metrics.inc("audio_samples_fed_to_streamer", samples_int16.size)
        metrics.set("streamer_buffer_samples", buf_after)
        self._dbg_calls += 1
        pending_out_total = sum(p.samples.size for p in self._pending_audio_for_output)
        metrics.set("pending_audio_for_output_samples", pending_out_total)
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
                self._video_queue_depth(),
                self._audio_out_buffer.size,
                pending_out_total,
            )
        if motion.shape[0] == 0:
            return
        self._last_motion_streamer_elapsed_s = streamer_elapsed_s
        motion_produced_at = time.perf_counter()
        if metrics.set_once("first_motion_s", motion_produced_at):
            snapshot = metrics.snapshot()["counters"]
            first_audio = snapshot.get("first_audio_push_s") or snapshot.get(
                "first_audio_process_s"
            )
            if first_audio is not None:
                metrics.set(
                    "first_motion_latency_s",
                    motion_produced_at - first_audio,
                )
        metrics.inc("motion_chunks_produced")
        metrics.inc("motion_frames_produced", motion.shape[0])
        metrics.set("last_motion_frames", motion.shape[0])
        metrics.set("last_motion_s", motion_produced_at)

        logger.warning(
            "[ARTalkPipeline] motion produced call#%d motion=%s",
            self._dbg_calls,
            tuple(motion.shape),
        )

        smoother_t0 = time.perf_counter()
        with torch.profiler.record_function("artalk.smoother_feed"):
            smoothed = self._smoother.feed(motion)
        smoother_elapsed_s = time.perf_counter() - smoother_t0
        metrics.observe_ms("smoother_feed", smoother_elapsed_s)
        metrics.observe_ms("post_model_smoother_feed", smoother_elapsed_s)
        metrics.inc("smoothed_frames_produced", smoothed.shape[0])
        metrics.set("last_smoothed_frames", smoothed.shape[0])
        if smoothed.shape[0] == 0:
            return
        # Couple output audio with frames that will actually be rendered. The
        # smoother can emit fewer frames than the raw model chunk, so using
        # smoothed.shape rather than motion.shape avoids releasing audio for
        # video frames that do not exist. Each frame maps to 640 samples at
        # 16 kHz / 25 fps.
        n_audio_samples_out = smoothed.shape[0] * SAMPLE_RATE // FPS
        audio_emit_t0 = time.perf_counter()
        emitted_audio_chunks = self._pop_pending_audio_for_output(n_audio_samples_out)
        emitted_audio_samples = sum(chunk.samples.size for chunk in emitted_audio_chunks)
        metrics.observe_ms(
            "post_model_audio_emit_prepare",
            time.perf_counter() - audio_emit_t0,
        )
        metrics.inc("audio_samples_emitted", emitted_audio_samples)
        metrics.set("last_audio_samples_emitted", emitted_audio_samples)
        render_chunk_t0 = time.perf_counter()
        rendered_in_chunk = 0
        segment_frames: list[RenderedVideoFrame] = []
        segment_render_started_at: float | None = None
        first_segment_render_started_at: float | None = None
        segment_index = 0
        segment_start_frame_in_chunk = 0

        def publish_pending_segment() -> None:
            nonlocal segment_frames, segment_render_started_at, segment_index
            if not segment_frames:
                return
            # Rendering a full ARTalk chunk can be slightly slower than the
            # chunk's 4.0 seconds of audio. Publishing only after the full chunk
            # caused audible gaps between chunks. Publishing around 1 second at
            # a time gives the output buffer chances to refill while the rest of
            # the chunk is still rendering, but keeps segments large enough to
            # avoid the previous per-render-batch crackle.
            segment_media_offset_s = 0.0
            segment_render_offset_s = 0.0
            segment_excess_s = 0.0
            if segment_render_started_at is not None:
                segment_render_offset_s = segment_render_started_at - motion_produced_at
                segment_media_offset_s = segment_start_frame_in_chunk / FPS
                segment_excess_s = max(
                    0.0,
                    segment_render_offset_s - segment_media_offset_s,
                )
            with pipeline_metrics_tags(
                pipeline_call=self._dbg_calls,
                segment_index=segment_index,
                segment_start_frame=segment_start_frame_in_chunk,
                segment_frames=len(segment_frames),
                segment_media_offset_s=segment_media_offset_s,
                segment_render_offset_s=segment_render_offset_s,
                segment_excess_s=segment_excess_s,
                audio_in_queue_depth=self._audio_in_queue.qsize(),
                video_queue_depth=self._video_queue_depth(),
                audio_out_buffer_s=self._audio_out_buffer.size / SAMPLE_RATE,
            ) as segment_metrics:
                n_audio_samples = len(segment_frames) * AUDIO_SAMPLES_PER_VIDEO_FRAME
                pair_t0 = time.perf_counter()
                audio_slice = self._consume_audio_chunks(
                    emitted_audio_chunks,
                    n_audio_samples,
                )
                segment_metrics.observe_ms(
                    "post_model_segment_audio_pairing",
                    time.perf_counter() - pair_t0,
                )
                segment_metrics.observe_ms(
                    "pre_render_wait_segment_start_offset",
                    segment_render_offset_s,
                )
                segment_metrics.observe_ms(
                    "pre_render_wait_segment_media_offset",
                    segment_media_offset_s,
                )
                segment_metrics.observe_ms(
                    "pre_render_wait_segment_excess_over_media",
                    segment_excess_s,
                )
                if first_segment_render_started_at is not None:
                    segment_metrics.observe_ms(
                        "pre_render_wait_prior_segment_backlog",
                        segment_render_started_at - first_segment_render_started_at,
                    )
                segment_metrics.set("last_pre_render_wait_segment_index", segment_index)
                segment_metrics.set(
                    "last_pre_render_wait_segment_start_frame",
                    segment_start_frame_in_chunk,
                )
                segment_metrics.set(
                    "last_pre_render_wait_segment_frames",
                    len(segment_frames),
                )
                segment_metrics.set(
                    "last_pre_render_wait_segment_render_offset_s",
                    segment_render_offset_s,
                )
                segment_metrics.set(
                    "last_pre_render_wait_segment_media_offset_s",
                    segment_media_offset_s,
                )
                segment_metrics.set(
                    "last_pre_render_wait_segment_excess_s",
                    segment_excess_s,
                )
                publish_t0 = time.perf_counter()
                with torch.profiler.record_function("artalk.publish_segment"):
                    self._publish_output_segment(
                        segment_frames,
                        audio_slice,
                        motion_produced_at,
                        segment_render_started_at,
                        segment_media_offset_s,
                    )
                segment_metrics.observe_ms(
                    "post_model_publish_segment",
                    time.perf_counter() - publish_t0,
                )
            segment_index += 1
            segment_frames = []
            segment_render_started_at = None

        for start in range(0, smoothed.shape[0], self._render_batch_size):
            motion_batch = smoothed[start : start + self._render_batch_size]
            render_t0 = time.perf_counter()
            if segment_render_started_at is None:
                segment_render_started_at = render_t0
                segment_start_frame_in_chunk = rendered_in_chunk
                if first_segment_render_started_at is None:
                    first_segment_render_started_at = render_t0
                    metrics.observe_ms(
                        "pre_render_wait_first_render_start",
                        first_segment_render_started_at - motion_produced_at,
                    )
            with pipeline_metrics_tags(
                pipeline_call=self._dbg_calls,
                segment_index=segment_index,
                segment_start_frame=segment_start_frame_in_chunk,
                render_batch_index=start // self._render_batch_size,
                render_start_frame=rendered_in_chunk,
                render_batch_frames=motion_batch.shape[0],
                audio_in_queue_depth=self._audio_in_queue.qsize(),
                video_queue_depth=self._video_queue_depth(),
                audio_out_buffer_s=self._audio_out_buffer.size / SAMPLE_RATE,
            ) as render_metrics:
                with torch.profiler.record_function("artalk.render_batch"):
                    rgb_batch, render_timings = self._renderer.render_batch_profile(
                        motion_batch
                    )
                for key, elapsed_s in render_timings.items():
                    render_metrics.observe_ms(key, elapsed_s)
                render_metrics.observe_ms(
                    "avatar_render_batch",
                    time.perf_counter() - render_t0,
                )
                render_metrics.inc("render_batches")
                render_metrics.inc("render_batch_frames", motion_batch.shape[0])
                convert_t0 = time.perf_counter()
                with torch.profiler.record_function("artalk.rgb_batch_to_numpy"):
                    arr_batch = (
                        (rgb_batch * 255.0)
                        .clamp_(0, 255)
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .numpy()
                    )
                render_metrics.observe_ms(
                    "rgb_batch_to_numpy",
                    time.perf_counter() - convert_t0,
                )
                render_finished_at = time.perf_counter()
            for arr in arr_batch:
                segment_frames.append(
                    RenderedVideoFrame(
                        image=arr,
                        motion_produced_at=motion_produced_at,
                        render_started_at=render_t0,
                        render_finished_at=render_finished_at,
                    )
                )
                metrics.inc("rendered_frames")
                rendered_in_chunk += 1
                metrics.set("last_rendered_s", time.perf_counter())
            if len(segment_frames) >= self._output_segment_min_frames:
                publish_pending_segment()
        publish_pending_segment()
        render_chunk_elapsed = time.perf_counter() - render_chunk_t0
        metrics.observe_ms("render_chunk_total", render_chunk_elapsed)
        metrics.set("last_render_chunk_frames", rendered_in_chunk)
        metrics.set("last_render_chunk_s", render_chunk_elapsed)
        render_chunk_media_s = rendered_in_chunk / FPS if rendered_in_chunk else 0.0
        metrics.set("last_render_chunk_media_s", render_chunk_media_s)
        if render_chunk_media_s > 0:
            metrics.set(
                "last_render_realtime_ratio",
                render_chunk_elapsed / render_chunk_media_s,
            )
        if render_chunk_elapsed > 0:
            metrics.set(
                "last_render_chunk_fps",
                rendered_in_chunk / render_chunk_elapsed,
            )
        logger.warning(
            "[ARTalkPipeline] frames rendered call#%d video_q=%d "
            "audio_out_buf=%d",
            self._dbg_calls,
            self._video_queue_depth(),
            self._audio_out_buffer.size,
        )

    def _video_queue_depth(self) -> int:
        with self._video_queue_lock:
            return len(self._video_queue)
