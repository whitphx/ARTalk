"""ARTalkPipeline output bookkeeping across flushes and turns (CPU only).

Run with ``python -m unittest tests.test_pipeline_output`` from the repo root.
"""

import time
import unittest
from unittest import mock

import numpy as np

from artalk import realtime_pipeline
from artalk.realtime_pipeline import AUDIO_SAMPLES_PER_VIDEO_FRAME, ARTalkPipeline
from artalk.streaming import CausalSavgolSmoother
from tests.test_blink_injector import RecordingRenderer, StubStreamer, fake_flame

FRAMES_PER_CHUNK = 100
CHUNK_SAMPLES = FRAMES_PER_CHUNK * AUDIO_SAMPLES_PER_VIDEO_FRAME
SMOOTHER_LAG_SAMPLES = CausalSavgolSmoother.DELAY * AUDIO_SAMPLES_PER_VIDEO_FRAME


def make_pipeline(renderer_cls=RecordingRenderer, **kwargs):
    with mock.patch.object(realtime_pipeline, "StreamingRenderer", renderer_cls):
        pipeline = ARTalkPipeline(
            streamer=StubStreamer(FRAMES_PER_CHUNK),
            flame_model=fake_flame(),
            mesh_renderer=None,
            device="cpu",
            renderer_stage_sync=False,
            warm_key_extra="output-test",
            **kwargs,
        )
    deadline = time.monotonic() + 30
    while pipeline._warm_key not in realtime_pipeline._WARMED_CONFIGS:
        if time.monotonic() > deadline:
            pipeline.stop()
            raise AssertionError("warm-up did not finish")
        time.sleep(0.01)
    return pipeline


class PendingAudioTest(unittest.TestCase):
    def test_turn_start_drops_audio_the_streamer_discards(self):
        pipeline = make_pipeline()
        pipeline._video_queue_high_water = 1 << 30
        gen = np.random.default_rng(0)
        accepted_at = 0.0
        try:
            with pipeline.metrics_context():
                for _ in range(6):
                    # Two and a half chunks, then a barge-in: the streamer
                    # holds half a chunk when the next turn starts.
                    for _ in range(5 * CHUNK_SAMPLES // 2 // 320):
                        samples = (3000 * gen.standard_normal(320)).astype(np.int16)
                        pipeline._process_sample_chunk(samples, accepted_at)
                        accepted_at += 0.005
                    pipeline.flush_output()
                    accepted_at += 2.0
                    pipeline._process_sample_chunk(np.zeros(320, np.int16), accepted_at)
                    pending = sum(c.samples.size for c in pipeline._pending_audio_for_output)
                    buffered = int(pipeline._streamer._audio_buffer.shape[0])
                    self.assertLessEqual(pending - buffered, SMOOTHER_LAG_SAMPLES)
        finally:
            pipeline.stop()


class SlowRenderer(RecordingRenderer):
    def render_batch_profile(self, motion):
        time.sleep(0.05)
        return super().render_batch_profile(motion)


def wait_for(condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


class DiscardInFlightTest(unittest.TestCase):
    def flush_mid_chunk(self, discard_in_flight):
        """Flush while the worker is rendering a chunk of silence, then push
        a second chunk; returns render batches after the flush and the
        pipeline, still running."""
        pipeline = make_pipeline(SlowRenderer, render_batch_size=8)
        pipeline._video_queue_high_water = 1 << 30
        renderer = pipeline._renderer
        renderer.frames.clear()

        def idle():
            snapshot = pipeline.output_buffer_snapshot()
            return not snapshot["worker_busy"] and snapshot["audio_in_queue_depth"] == 0

        pipeline.push_silence(CHUNK_SAMPLES / realtime_pipeline.SAMPLE_RATE)
        # Past the first published segment, so the flush has output to drop.
        wait_for(lambda: len(renderer.frames) >= 5)
        pipeline.flush_output(discard_in_flight=discard_in_flight)
        batches_at_flush = len(renderer.frames)
        wait_for(idle)
        batches_after_flush = len(renderer.frames) - batches_at_flush
        return batches_after_flush, pipeline

    def test_flush_stops_the_chunk_in_flight(self):
        batches_after_flush, pipeline = self.flush_mid_chunk(True)
        try:
            self.assertLessEqual(batches_after_flush, 1)
            snapshot = pipeline.output_buffer_snapshot()
            self.assertEqual(snapshot["video_queue_depth"], 0)
            self.assertEqual(snapshot["audio_out_buffer_samples"], 0)
            # Audio pushed after the flush renders and publishes in full, on
            # frame indices the media clock reaches.
            pipeline.push_silence(CHUNK_SAMPLES / realtime_pipeline.SAMPLE_RATE)
            wait_for(lambda: pipeline.output_buffer_snapshot()["video_queue_depth"] > 0)
            wait_for(lambda: not pipeline.output_buffer_snapshot()["worker_busy"])
            snapshot = pipeline.output_buffer_snapshot()
            self.assertEqual(snapshot["video_queue_depth"], FRAMES_PER_CHUNK)
            self.assertEqual(
                snapshot["audio_out_buffer_samples"],
                FRAMES_PER_CHUNK * AUDIO_SAMPLES_PER_VIDEO_FRAME,
            )
            self.assertEqual(
                pipeline._video_queue[0][0],
                pipeline._synced_audio_samples_served // AUDIO_SAMPLES_PER_VIDEO_FRAME,
            )
        finally:
            pipeline.stop()

    def test_plain_flush_lets_the_chunk_in_flight_publish(self):
        batches_after_flush, pipeline = self.flush_mid_chunk(False)
        try:
            self.assertGreater(batches_after_flush, 1)
            self.assertGreater(pipeline.output_buffer_snapshot()["video_queue_depth"], 0)
        finally:
            pipeline.stop()


if __name__ == "__main__":
    unittest.main()
