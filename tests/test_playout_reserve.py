"""Playout reserve steering in ARTalkPipeline.audio_source_callback (CPU only).

Run with ``python -m unittest tests.test_playout_reserve`` from the repo root.
"""

import fractions
import time
import unittest
from unittest import mock

import numpy as np

from artalk import realtime_pipeline
from artalk.realtime_pipeline import (
    AUDIO_OUT_PTIME,
    AUDIO_OUT_SAMPLES_PER_FRAME,
    PLAYOUT_RESERVE_TOLERANCE_S,
    SAMPLE_RATE,
)
from tests.test_pipeline_output import CHUNK_SAMPLES, make_pipeline

N = AUDIO_OUT_SAMPLES_PER_FRAME
TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)
TOLERANCE = int(SAMPLE_RATE * PLAYOUT_RESERVE_TOLERANCE_S)


class RealtimeProducer:
    """Pushes input at realtime pace, one frame per playback callback, on a
    simulated clock. It also stands in for the worker: queued input moves to
    the output buffer while the buffer is below ``max_depth``, the way
    rendering waits at the video queue's high-water mark."""

    def __init__(self, pipeline, source, real=True, max_depth=None):
        self.pipeline = pipeline
        self.source = source
        self.real = real
        self.max_depth = max_depth
        self.pos = 0
        self.now = 1000.0
        self.played = []
        self.depths = []

    def arrive(self, n_samples):
        chunk = self.source[self.pos : self.pos + n_samples]
        self.pos += chunk.size
        self.pipeline.push_audio_samples(chunk, is_filler=not self.real)

    def deliver(self):
        pipeline = self.pipeline
        while not pipeline._audio_in_queue.empty() and (
            self.max_depth is None or pipeline._audio_out_buffer.size < self.max_depth
        ):
            item = pipeline._audio_in_queue.get_nowait()
            if not item.is_filler:
                # What the worker records for each piece of real input.
                pipeline._last_real_accepted_at = item.accepted_at
            with pipeline._audio_out_lock:
                pipeline._audio_out_buffer = np.concatenate(
                    [pipeline._audio_out_buffer, item.samples]
                )

    def push(self, n_samples):
        self.arrive(n_samples)
        self.deliver()

    def run(self, n_callbacks, arriving=True):
        for _ in range(n_callbacks):
            if arriving:
                self.arrive(N)
            self.deliver()
            pts = len(self.played) * N
            frame = self.pipeline.audio_source_callback(pts, TIME_BASE)
            self.now += AUDIO_OUT_PTIME
            self.played.append(frame.to_ndarray().reshape(-1))
            self.depths.append(self.pipeline._audio_out_buffer.size)
        return np.concatenate(self.played)

    def reserve(self):
        """The buffer's low point over the last model chunk."""
        return min(self.depths[-CHUNK_SAMPLES // N :])

    def backlog(self):
        queued = sum(item.samples.size for item in self.pipeline._audio_in_queue.queue)
        return self.pipeline._audio_out_buffer.size + queued


def playing_pipeline(reserve_seconds):
    pipeline = make_pipeline(
        output_audio_prebuffer_seconds=0.0,
        output_playout_reserve_seconds=reserve_seconds,
    )
    pipeline._video_queue_high_water = 1 << 30
    # RealtimeProducer does the worker's part, so that input can wait in the
    # queue.
    pipeline._stop_event.set()
    pipeline._worker_thread.join()
    pipeline._stop_event.clear()
    return pipeline


def speech(n_samples, seed=0):
    gen = np.random.default_rng(seed)
    return (3000 * gen.standard_normal(n_samples)).astype(np.int16)


class PlayoutReserveTest(unittest.TestCase):
    def producer(self, pipeline, source, **kwargs):
        producer = RealtimeProducer(pipeline, source, **kwargs)
        clock = mock.Mock(wraps=time)
        clock.perf_counter = lambda: producer.now
        patcher = mock.patch.object(realtime_pipeline, "time", clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        return producer

    def run_silence(self, start_depth_s, reserve_s, seconds=20, real=True):
        pipeline = playing_pipeline(reserve_s)
        producer = self.producer(pipeline, np.zeros(SAMPLE_RATE * 60, np.int16), real=real)
        try:
            with pipeline.metrics_context():
                producer.push(int(SAMPLE_RATE * start_depth_s))
                producer.run(int(seconds / 0.02))
                with pipeline._audio_out_lock:
                    clock = pipeline._synced_audio_samples_served
                    buffered = pipeline._audio_out_buffer.size
            return producer, clock, buffered
        finally:
            pipeline.stop()

    def test_silence_above_the_target_is_trimmed_to_it(self):
        producer, clock, buffered = self.run_silence(1.5, 0.6)
        self.assertLessEqual(abs(producer.reserve() - int(0.6 * SAMPLE_RATE)), TOLERANCE + N)
        # Dropped silence is credited to the media clock, so the video
        # frame for every sample still left in the buffer stays reachable.
        self.assertEqual(clock + buffered, producer.pos)

    def test_silence_below_the_target_is_stretched_to_it(self):
        producer, clock, buffered = self.run_silence(0.2, 0.6)
        self.assertLessEqual(abs(producer.reserve() - int(0.6 * SAMPLE_RATE)), TOLERANCE + N)
        self.assertEqual(clock + buffered, producer.pos)

    def test_stretching_stops_at_the_latency_budget(self):
        pipeline = playing_pipeline(0.6)
        pipeline.set_output_underrun_policy("continuous", max_added_latency_seconds=0.1)
        producer = self.producer(pipeline, np.zeros(SAMPLE_RATE * 60, np.int16))
        try:
            with pipeline.metrics_context():
                producer.push(int(0.2 * SAMPLE_RATE))
                producer.run(int(20 / 0.02))
        finally:
            pipeline.stop()
        self.assertEqual(producer.depths[-1], int(0.3 * SAMPLE_RATE))

    def test_a_backlog_after_a_delivery_gap_is_trimmed(self):
        # The producer stalls, then sends what it held back all at once,
        # more than rendering takes in before it waits at the high-water
        # mark. Playback is then that far behind a stream still arriving at
        # realtime pace, and the worker is seconds behind in the input queue.
        pipeline = playing_pipeline(0.6)
        producer = self.producer(
            pipeline, np.zeros(SAMPLE_RATE * 150, np.int16), max_depth=6 * SAMPLE_RATE
        )
        try:
            with pipeline.metrics_context():
                producer.push(int(0.6 * SAMPLE_RATE))
                producer.run(int(10 / AUDIO_OUT_PTIME))
                producer.run(int(12 / AUDIO_OUT_PTIME), arriving=False)
                for _ in range(120):
                    producer.arrive(SAMPLE_RATE // 10)
                producer.run(int(80 / AUDIO_OUT_PTIME))
                backlog = producer.backlog()
        finally:
            pipeline.stop()
        self.assertLessEqual(abs(backlog - int(0.6 * SAMPLE_RATE)), TOLERANCE + N)

    def test_idle_filler_keeps_the_depth(self):
        producer, _, _ = self.run_silence(1.5, 0.6, real=False)
        self.assertEqual(producer.reserve(), int(1.5 * SAMPLE_RATE))

    def test_no_target_keeps_the_depth(self):
        producer, _, _ = self.run_silence(1.5, None)
        self.assertEqual(producer.reserve(), int(1.5 * SAMPLE_RATE))

    def test_speech_is_played_untouched(self):
        pipeline = playing_pipeline(0.2)
        source = speech(SAMPLE_RATE * 30)
        producer = self.producer(pipeline, source)
        try:
            with pipeline.metrics_context():
                producer.push(SAMPLE_RATE)
                played = producer.run(int(10 / 0.02))
        finally:
            pipeline.stop()
        np.testing.assert_array_equal(played, source[: played.size])
        self.assertEqual(producer.reserve(), SAMPLE_RATE)

    def test_pauses_between_words_are_kept(self):
        # Words 0.5 s long with 0.25 s gaps never give steering a silence
        # long enough to act in.
        word, gap = speech(SAMPLE_RATE // 2), np.zeros(SAMPLE_RATE // 4, np.int16)
        source = np.concatenate([word, gap] * 40)
        pipeline = playing_pipeline(0.2)
        producer = self.producer(pipeline, source)
        try:
            with pipeline.metrics_context():
                producer.push(SAMPLE_RATE)
                played = producer.run(int(20 / 0.02))
        finally:
            pipeline.stop()
        np.testing.assert_array_equal(played, source[: played.size])

    def test_a_silence_after_speech_trims_only_the_silence(self):
        source = np.concatenate(
            [speech(SAMPLE_RATE * 3), np.zeros(SAMPLE_RATE * 6, np.int16), speech(SAMPLE_RATE * 3, 1)]
        )
        pipeline = playing_pipeline(0.4)
        producer = self.producer(pipeline, source)
        try:
            with pipeline.metrics_context():
                producer.push(int(1.2 * SAMPLE_RATE))
                played = producer.run(int(10 / 0.02))
        finally:
            pipeline.stop()
        voiced = played[np.abs(played) > 0]
        expected = source[np.abs(source) > 0]
        np.testing.assert_array_equal(voiced, expected[: voiced.size])
        # The second utterance plays with the trimmed depth ahead of it.
        self.assertLessEqual(abs(producer.depths[-1] - int(0.4 * SAMPLE_RATE)), TOLERANCE + N)


if __name__ == "__main__":
    unittest.main()
