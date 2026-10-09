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


if __name__ == "__main__":
    unittest.main()
