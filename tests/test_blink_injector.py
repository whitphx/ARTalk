"""BlinkInjector unit tests and its wiring in ARTalkPipeline (CPU only).

Run with ``python -m unittest tests.test_blink_injector`` from the repo root.
Set ``ARTALK_FLAME_MODEL`` to a ``FLAME_with_eye.pt`` to also check closure
depth on the real FLAME model.
"""

import os
import random
import time
import types
import unittest
from unittest import mock

import numpy as np
import torch

from artalk import realtime_pipeline
from artalk.realtime_pipeline import ARTalkPipeline, BlinkInjector, SilentMouthGate

FPS = 25
MOUTH = list(SilentMouthGate.MOUTH_DIMS)


def fake_flame():
    """A FLAME stand-in whose eyelids open 8 mm and close along expression
    dim 4, the largest non-mouth component of the blink direction."""
    v_template = torch.zeros(5023, 3)
    shapedirs = torch.zeros(5023, 3, 400)
    for upper, lower in BlinkInjector.EYELID_PAIRS:
        v_template[upper, 1] = 0.004
        v_template[lower, 1] = -0.004
        shapedirs[upper, 1, 304] = 0.002
        shapedirs[lower, 1, 304] = -0.002
    return types.SimpleNamespace(v_template=v_template, shapedirs=shapedirs)


def run(injector, motion, chunk):
    return torch.cat(
        [injector.process(motion[s : s + chunk]) for s in range(0, motion.shape[0], chunk)]
    )


def onsets(injected, motion):
    """Frames where an injected blink starts."""
    active = (injected - motion).abs().sum(-1) > 0
    return np.flatnonzero(active[1:] & ~active[:-1]) + 1


class BlinkInjectorTest(unittest.TestCase):
    def setUp(self):
        self.flame = fake_flame()
        gen = torch.Generator().manual_seed(0)
        self.motion = 0.05 * torch.randn(60 * FPS, 106, generator=gen)

    def test_mouth_and_pose_dims_bit_identical(self):
        out = run(BlinkInjector.from_flame(self.flame, seed=1), self.motion, 25)
        self.assertGreater(len(onsets(out, self.motion)), 0)
        self.assertTrue(torch.equal(out[:, MOUTH], self.motion[:, MOUTH]))
        self.assertTrue(torch.equal(out[:, 100:], self.motion[:, 100:]))
        self.assertFalse(torch.equal(out, self.motion))

    def test_deterministic_under_seed(self):
        a = run(BlinkInjector.from_flame(self.flame, seed=3), self.motion, 25)
        b = run(BlinkInjector.from_flame(self.flame, seed=3), self.motion, 25)
        c = run(BlinkInjector.from_flame(self.flame, seed=4), self.motion, 25)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_independent_of_chunk_size(self):
        # 25 and 100 frames are the 1 s and 4 s streamers' chunk sizes; 7
        # and 1 cover the smoother's uneven output.
        ref = run(BlinkInjector.from_flame(self.flame, seed=5), self.motion, 25)
        for chunk in (100, 7, 1):
            out = run(BlinkInjector.from_flame(self.flame, seed=5), self.motion, chunk)
            self.assertTrue(torch.equal(out, ref), chunk)

    def test_rate_matches_target(self):
        # The interval distribution is heavy-tailed, so one 2 h run still
        # varies by several percent; average a few.
        motion = torch.zeros(120 * 60 * FPS, 106)
        for rate in (BlinkInjector.RATE_PER_MIN, 10.0):
            injector_cls = type("RateInjector", (BlinkInjector,), {"RATE_PER_MIN": rate})
            measured = np.mean([
                len(onsets(run(injector_cls.from_flame(self.flame, seed=s), motion, 100), motion)) / 120
                for s in range(4)
            ])
            self.assertAlmostEqual(measured / rate, 1.0, delta=0.05, msg=f"{measured} vs {rate}")

    def test_blink_closes_and_reopens(self):
        motion = torch.zeros(30 * FPS, 106)
        injector = BlinkInjector.from_flame(self.flame, seed=8)
        out = run(injector, motion, 25)
        aperture = injector.aperture(out).numpy()
        self.assertLess(aperture.min(), 0.5 * aperture.max())

    def test_no_blink_starts_during_model_blink(self):
        motion = torch.zeros(10 * 60 * FPS, 106)
        dips = []
        for start in range(FPS, motion.shape[0], FPS):
            motion[start : start + 6, 4] = -1.5
            dips.extend(range(start, start + 6))
        out = run(BlinkInjector.from_flame(self.flame, seed=9), motion, 25)
        self.assertFalse(set(onsets(out, motion)) & set(dips))

    def model_blinks_every(self, seconds, minutes):
        motion = torch.zeros(minutes * 60 * FPS, 106)
        for start in range(round(seconds * FPS), motion.shape[0], round(seconds * FPS)):
            motion[start : start + 6, 4] = -1.5
        out = run(BlinkInjector.from_flame(self.flame, seed=12), motion, 25)
        return len(onsets(out, motion)) / minutes

    def test_model_blinks_count_toward_rate(self):
        target = BlinkInjector.RATE_PER_MIN
        # A model blinking above the target gets little added: only bursts of
        # short intervals that outrun the credit cap leak through.
        self.assertLess(self.model_blinks_every(3.0, 20), 0.3 * target)
        # At half the target the injector supplies roughly the other half.
        injected = self.model_blinks_every(120 / target, 60)
        self.assertAlmostEqual(injected / (target / 2), 1.0, delta=0.25, msg=injected)

    def injected_per_min(self, model_blinking_first, window_s, seeds=range(20)):
        """Injected rate in ``window_s`` (seconds after the switch) when the
        model blinks at 20/min for 90 s and then stops, or the reverse."""
        motion = torch.zeros(150 * FPS, 106)
        lo, hi = (0, 90 * FPS) if model_blinking_first else (90 * FPS, 150 * FPS)
        for start in range(lo + 3 * FPS, hi, 3 * FPS):
            motion[start : start + 6, 4] = -1.5
        count = 0
        for seed in seeds:
            out = run(BlinkInjector.from_flame(self.flame, seed=seed), motion, 25)
            t = onsets(out, motion) / FPS - 90
            count += int(((t >= window_s[0]) & (t < window_s[1])).sum())
        return count / len(seeds) / ((window_s[1] - window_s[0]) / 60)

    def test_injection_resumes_when_model_stops_blinking(self):
        # At most MAX_CREDITS scheduled blinks are skipped once the model goes quiet.
        target = BlinkInjector.RATE_PER_MIN
        self.assertGreater(self.injected_per_min(True, (0, 20)), 0.5 * target)
        self.assertGreater(self.injected_per_min(True, (20, 60)), 0.8 * target)

    def test_injection_stops_when_model_starts_blinking(self):
        target = BlinkInjector.RATE_PER_MIN
        self.assertLess(self.injected_per_min(False, (10, 60)), 0.3 * target)

    def test_global_rng_untouched(self):
        torch.manual_seed(11)
        np.random.seed(11)
        random.seed(11)
        states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        run(BlinkInjector.from_flame(self.flame), self.motion, 25)
        self.assertTrue(torch.equal(torch.get_rng_state(), states[0]))
        np_state = np.random.get_state()
        self.assertEqual(np_state[0], states[1][0])
        self.assertTrue(np.array_equal(np_state[1], states[1][1]))
        self.assertEqual(np_state[2:], states[1][2:])
        self.assertEqual(random.getstate(), states[2])

    @unittest.skipUnless(os.environ.get("ARTALK_FLAME_MODEL"), "ARTALK_FLAME_MODEL not set")
    def test_overlap_closes_no_further_than_alone_on_flame(self):
        from artalk.flame_model.FLAME import FLAMEModel

        flame = FLAMEModel(n_shape=300, n_exp=100, no_lmks=True,
                           model_path=os.environ["ARTALK_FLAME_MODEL"])
        injector = BlinkInjector.from_flame(flame)
        direction = torch.tensor(BlinkInjector.BLINK_DIRECTION)
        open_frame = torch.zeros(1, 106)
        open_mm = injector.aperture(open_frame).item()
        step = torch.zeros(1, 106)
        step[0, injector._dims] = injector._direction
        shut_mm = injector.aperture(open_frame + step).item()
        lids = injector._lids(open_frame).numpy()[0]
        self.assertEqual(injector._limit_overlap(lids, open_mm, 1.0), 1.0)
        # The model's own blink at several depths (the corpus mean blink, all dims).
        for depth in (0.3, 0.6, 1.0, 1.5):
            frame = torch.zeros(1, 106)
            frame[0, :100] = depth * direction
            weight = injector._limit_overlap(injector._lids(frame).numpy()[0], open_mm, 1.0)
            combined = injector.aperture(frame + weight * step).item()
            self.assertGreaterEqual(combined, shut_mm - 1e-3, depth)
            self.assertLess(weight, 1.0, depth)

    @unittest.skipUnless(os.environ.get("ARTALK_FLAME_MODEL"), "ARTALK_FLAME_MODEL not set")
    def test_closure_on_flame(self):
        from artalk.flame_model.FLAME import FLAMEModel

        flame = FLAMEModel(n_shape=300, n_exp=100, no_lmks=True,
                           model_path=os.environ["ARTALK_FLAME_MODEL"])
        motion = torch.zeros(30 * FPS, 106)
        injector = BlinkInjector.from_flame(flame, seed=10)
        aperture = injector.aperture(run(injector, motion, 25)).numpy()
        # Full closure on the mesh reads at about 0.4 of the open aperture.
        self.assertLess(aperture.min() / aperture.max(), 0.55)


class RecordingRenderer:
    mode = "mesh"

    def __init__(self, **kwargs):
        self.frames = []

    def render_batch_profile(self, motion):
        self.frames.append(motion.detach().clone())
        return torch.zeros(motion.shape[0], 4, 4, 3, dtype=torch.uint8), {}


class StubStreamer:
    """Emits fixed pseudo-random motion one whole chunk at a time."""

    motion_dim = 106

    def __init__(self, frames_per_chunk):
        self.frames_per_chunk = frames_per_chunk
        self.patch_audio_length = frames_per_chunk * realtime_pipeline.AUDIO_SAMPLES_PER_VIDEO_FRAME
        self.reset()

    def reset(self):
        self._audio_buffer = torch.zeros(0)
        self._gen = torch.Generator().manual_seed(0)

    def feed(self, samples):
        self._audio_buffer = torch.cat([self._audio_buffer, samples.float().cpu()])
        n = self._audio_buffer.shape[0] // self.patch_audio_length
        self._audio_buffer = self._audio_buffer[n * self.patch_audio_length :]
        return 0.3 * torch.randn(n * self.frames_per_chunk, 106, generator=self._gen)


class PipelineWiringTest(unittest.TestCase):
    def rendered_motion(self, frames_per_chunk, **kwargs):
        with mock.patch.object(realtime_pipeline, "StreamingRenderer", RecordingRenderer):
            pipeline = ARTalkPipeline(
                streamer=StubStreamer(frames_per_chunk),
                flame_model=fake_flame(),
                mesh_renderer=None,
                device="cpu",
                renderer_stage_sync=False,
                warm_key_extra=f"blink-test-{frames_per_chunk}",
                **kwargs,
            )
        try:
            deadline = time.monotonic() + 30
            while pipeline._warm_key not in realtime_pipeline._WARMED_CONFIGS:
                self.assertLess(time.monotonic(), deadline, "warm-up did not finish")
                time.sleep(0.01)
            pipeline._video_queue_high_water = 1 << 30
            pipeline._renderer.frames.clear()
            gen = np.random.default_rng(0)
            # Alternating 2 s of speech-level noise and 2 s of silence, in
            # the 20 ms items the realtime path delivers, so the gate engages.
            with pipeline.metrics_context():
                for second in range(16):
                    level = 3000 if (second // 2) % 2 == 0 else 0
                    for _ in range(50):
                        samples = (level * gen.standard_normal(320)).astype(np.int16)
                        pipeline._process_sample_chunk(samples, time.perf_counter())
            return torch.cat(pipeline._renderer.frames)
        finally:
            pipeline.stop()

    def test_injector_leaves_mouth_dims_bit_identical(self):
        for frames_per_chunk in (25, 100):
            off = self.rendered_motion(frames_per_chunk)
            on = self.rendered_motion(frames_per_chunk, blink_injector=True, blink_seed=0)
            self.assertEqual(off.shape, on.shape)
            self.assertTrue(torch.equal(on[:, MOUTH], off[:, MOUTH]), frames_per_chunk)
            self.assertTrue(torch.equal(on[:, 100:], off[:, 100:]), frames_per_chunk)
            self.assertFalse(torch.equal(on, off), frames_per_chunk)

    def test_flag_off_builds_no_injector(self):
        with mock.patch.object(realtime_pipeline, "StreamingRenderer", RecordingRenderer):
            pipeline = ARTalkPipeline(
                streamer=StubStreamer(25), flame_model=fake_flame(), mesh_renderer=None,
                device="cpu", renderer_stage_sync=False,
            )
        pipeline.stop()
        self.assertIsNone(pipeline._blink_injector)


if __name__ == "__main__":
    unittest.main()
