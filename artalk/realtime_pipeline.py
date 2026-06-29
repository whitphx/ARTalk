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
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import av
import numpy as np
import torch

from .metrics import (
    PipelineMetrics,
    current_pipeline_metrics,
    observe_pipeline_duration,
    pipeline_metrics_context,
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


@dataclass
class ConsumedAudioSlice:
    samples: np.ndarray
    first_accepted_at: float | None
    midpoint_accepted_at: float | None


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
        # Video frames are indexed by ARTalk's 25 fps motion clock, not by
        # WebRTC callback count. The audio source callback advances the media
        # clock using the number of real delayed samples it has emitted; the
        # video source callback then serves the newest frame whose index is not
        # ahead of that audio clock. This prevents video from racing ahead when
        # rendering completes before enough delayed audio has actually played.
        self._video_queue: deque[tuple[int, np.ndarray]] = deque()
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
        arr, frame_index = self._pop_video_frame_for_media_clock(target_index)
        metrics.set("last_video_target_frame_index", target_index)
        if arr is None:
            arr = self._placeholder
            metrics.inc("video_placeholder_frames")
            metrics.inc("video_no_ready_frame_callbacks")
        else:
            self._placeholder = arr
            self._last_video_frame_index_served = frame_index
            metrics.inc("video_frames_served")
            metrics.set("last_video_frame_index_served", frame_index)
            self._record_served_video_lag(target_index, frame_index)
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
    ) -> tuple[np.ndarray | None, int | None]:
        # The browser asks for video at a steady cadence even when rendering is
        # bursty. If rendering fell behind and then catches up, multiple queued
        # frames can be older than the audio clock. Drop those stale frames and
        # serve the latest frame at or before the current audio-derived index.
        dropped_for_sync = 0
        with self._video_queue_lock:
            selected: tuple[int, np.ndarray] | None = None
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

    def _pop_pending_audio_for_output(self, n_samples: int) -> list[PendingAudioChunk]:
        # ARTalk consumes audio in large model chunks before it can produce
        # motion. Keep the raw input audio staged until the corresponding
        # smoothed motion frames are known, then pop exactly the samples that
        # map to those rendered frames. Chunks keep their original accept
        # timestamp so segment publication can report end-to-end latency without
        # keeping a full event timeline.
        remaining = max(0, n_samples)
        emitted: list[PendingAudioChunk] = []
        while remaining and self._pending_audio_for_output:
            chunk = self._pending_audio_for_output.pop(0)
            if chunk.samples.size <= remaining:
                emitted.append(chunk)
                remaining -= chunk.samples.size
                continue
            emitted.append(
                PendingAudioChunk(
                    samples=chunk.samples[:remaining],
                    accepted_at=chunk.accepted_at,
                )
            )
            self._pending_audio_for_output.insert(
                0,
                PendingAudioChunk(
                    samples=chunk.samples[remaining:],
                    accepted_at=chunk.accepted_at,
                ),
            )
            remaining = 0
        pending_samples = sum(chunk.samples.size for chunk in self._pending_audio_for_output)
        metrics = current_pipeline_metrics()
        metrics.set("pending_audio_for_output_samples", pending_samples)
        return emitted

    @staticmethod
    def _consume_audio_chunks(
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
            consumed.append(chunk.samples[:remaining])
            consumed_chunks.append(
                PendingAudioChunk(
                    samples=chunk.samples[:remaining],
                    accepted_at=chunk.accepted_at,
                )
            )
            chunks.insert(
                0,
                PendingAudioChunk(
                    samples=chunk.samples[remaining:],
                    accepted_at=chunk.accepted_at,
                ),
            )
            remaining = 0
        if not consumed:
            return ConsumedAudioSlice(
                samples=np.zeros(0, dtype=np.int16),
                first_accepted_at=accepted_at,
                midpoint_accepted_at=accepted_at,
            )
        midpoint_accepted_at = accepted_at
        midpoint_sample = sum(chunk.samples.size for chunk in consumed_chunks) // 2
        cursor = 0
        for chunk in consumed_chunks:
            next_cursor = cursor + chunk.samples.size
            if midpoint_sample < next_cursor:
                midpoint_accepted_at = chunk.accepted_at
                break
            cursor = next_cursor
        return ConsumedAudioSlice(
            samples=np.concatenate(consumed),
            first_accepted_at=accepted_at,
            midpoint_accepted_at=midpoint_accepted_at,
        )

    def _publish_output_segment(
        self,
        frames: list[np.ndarray],
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
        publish_video_t0 = time.perf_counter()
        with self._video_queue_lock:
            for arr in frames:
                frame_index = self._next_video_frame_index
                self._next_video_frame_index += 1
                while len(self._video_queue) >= self._video_queue_max:
                    self._video_queue.popleft()
                    dropped_video_frames += 1
                self._video_queue.append((frame_index, arr))
                last_frame_index = frame_index
            video_depth = len(self._video_queue)
        metrics.observe_ms(
            "post_model_publish_video_queue",
            time.perf_counter() - publish_video_t0,
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
            pre_model_compute_s = self._last_motion_streamer_elapsed_s
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
                except Exception:
                    logger.exception("[ARTalkPipeline] worker error")
                finally:
                    self._worker_busy = False
                    metrics.set("worker_busy", 0)

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
        segment_frames: list[np.ndarray] = []
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
            n_audio_samples = len(segment_frames) * AUDIO_SAMPLES_PER_VIDEO_FRAME
            pair_t0 = time.perf_counter()
            audio_slice = self._consume_audio_chunks(
                emitted_audio_chunks,
                n_audio_samples,
            )
            metrics.observe_ms(
                "post_model_segment_audio_pairing",
                time.perf_counter() - pair_t0,
            )
            segment_media_offset_s = 0.0
            if segment_render_started_at is not None:
                segment_render_offset_s = segment_render_started_at - motion_produced_at
                segment_media_offset_s = segment_start_frame_in_chunk / FPS
                segment_excess_s = max(
                    0.0,
                    segment_render_offset_s - segment_media_offset_s,
                )
                metrics.observe_ms(
                    "pre_render_wait_segment_start_offset",
                    segment_render_offset_s,
                )
                metrics.observe_ms(
                    "pre_render_wait_segment_media_offset",
                    segment_media_offset_s,
                )
                metrics.observe_ms(
                    "pre_render_wait_segment_excess_over_media",
                    segment_excess_s,
                )
                if first_segment_render_started_at is not None:
                    metrics.observe_ms(
                        "pre_render_wait_prior_segment_backlog",
                        segment_render_started_at - first_segment_render_started_at,
                    )
                metrics.set("last_pre_render_wait_segment_index", segment_index)
                metrics.set(
                    "last_pre_render_wait_segment_start_frame",
                    segment_start_frame_in_chunk,
                )
                metrics.set(
                    "last_pre_render_wait_segment_frames",
                    len(segment_frames),
                )
                metrics.set(
                    "last_pre_render_wait_segment_render_offset_s",
                    segment_render_offset_s,
                )
                metrics.set(
                    "last_pre_render_wait_segment_media_offset_s",
                    segment_media_offset_s,
                )
                metrics.set(
                    "last_pre_render_wait_segment_excess_s",
                    segment_excess_s,
                )
            publish_t0 = time.perf_counter()
            self._publish_output_segment(
                segment_frames,
                audio_slice,
                motion_produced_at,
                segment_render_started_at,
                segment_media_offset_s,
            )
            metrics.observe_ms(
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
            rgb_batch, render_timings = self._renderer.render_batch_profile(
                motion_batch
            )
            for key, elapsed_s in render_timings.items():
                metrics.observe_ms(key, elapsed_s)
            metrics.observe_ms(
                "avatar_render_batch",
                time.perf_counter() - render_t0,
            )
            metrics.inc("render_batches")
            metrics.inc("render_batch_frames", motion_batch.shape[0])
            convert_t0 = time.perf_counter()
            arr_batch = (
                (rgb_batch * 255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .numpy()
            )
            metrics.observe_ms(
                "rgb_batch_to_numpy",
                time.perf_counter() - convert_t0,
            )
            for arr in arr_batch:
                segment_frames.append(arr)
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
