"""Playout reserve steering in ARTalkPipeline.audio_source_callback (CPU only).

Run with ``python -m unittest tests.test_playout_reserve`` from the repo root.
"""

import fractions
import unittest

import numpy as np

from artalk.realtime_pipeline import (
    AUDIO_OUT_SAMPLES_PER_FRAME,
    PLAYOUT_RESERVE_TOLERANCE_S,
    SAMPLE_RATE,
)
from tests.test_pipeline_output import CHUNK_SAMPLES, make_pipeline

N = AUDIO_OUT_SAMPLES_PER_FRAME
TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)
TOLERANCE = int(SAMPLE_RATE * PLAYOUT_RESERVE_TOLERANCE_S)


class RealtimeProducer:
    """Feeds the output buffer one frame per callback, the way a producer
    streaming at realtime pace keeps the depth it started with."""

    def __init__(self, pipeline, source):
        self.pipeline = pipeline
        self.source = source
        self.pos = 0
        self.played = []
        self.depths = []

    def push(self, n_samples):
        chunk = self.source[self.pos : self.pos + n_samples]
        self.pos += chunk.size
        with self.pipeline._audio_out_lock:
            self.pipeline._audio_out_buffer = np.concatenate(
                [self.pipeline._audio_out_buffer, chunk]
            )

    def run(self, n_callbacks):
        for i in range(n_callbacks):
            self.push(N)
            frame = self.pipeline.audio_source_callback(i * N, TIME_BASE)
            self.played.append(frame.to_ndarray().reshape(-1))
            self.depths.append(self.pipeline._audio_out_buffer.size)
        return np.concatenate(self.played)

    def reserve(self):
        """The buffer's low point over the last model chunk."""
        return min(self.depths[-CHUNK_SAMPLES // N :])


def playing_pipeline(reserve_seconds):
    pipeline = make_pipeline(
        output_audio_prebuffer_seconds=0.0,
        output_playout_reserve_seconds=reserve_seconds,
    )
    pipeline._video_queue_high_water = 1 << 30
    return pipeline


def speech(n_samples, seed=0):
    gen = np.random.default_rng(seed)
    return (3000 * gen.standard_normal(n_samples)).astype(np.int16)


class PlayoutReserveTest(unittest.TestCase):
    def run_silence(self, start_depth_s, reserve_s, seconds=20):
        pipeline = playing_pipeline(reserve_s)
        producer = RealtimeProducer(pipeline, np.zeros(SAMPLE_RATE * 60, np.int16))
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
        producer = RealtimeProducer(pipeline, np.zeros(SAMPLE_RATE * 60, np.int16))
        try:
            with pipeline.metrics_context():
                producer.push(int(0.2 * SAMPLE_RATE))
                producer.run(int(20 / 0.02))
        finally:
            pipeline.stop()
        self.assertEqual(producer.depths[-1], int(0.3 * SAMPLE_RATE))

    def test_no_target_keeps_the_depth(self):
        producer, _, _ = self.run_silence(1.5, None)
        self.assertEqual(producer.reserve(), int(1.5 * SAMPLE_RATE))

    def test_speech_is_played_untouched(self):
        pipeline = playing_pipeline(0.2)
        source = speech(SAMPLE_RATE * 30)
        producer = RealtimeProducer(pipeline, source)
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
        producer = RealtimeProducer(pipeline, source)
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
        producer = RealtimeProducer(pipeline, source)
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
