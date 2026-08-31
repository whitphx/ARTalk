# Frame-Level Streaming — Scaffold Assessment and Experiment Spec

Status: assessment of existing work plus a proposed experiment. The scaffold
under review is uncommitted work-in-progress on the `realtime` branch of the
main ARTalk checkout (`artalk_frame/`, `train_code/core/models/artalk_stream/`,
`train_code/core/trainer/stream_trainer.py`, `configs/artalk_stream.yaml`,
`app/frame_streaming.py`, `scripts/check_frame_model.py`,
`docs/frame_model.md`, `tests/test_frame_model.py`). Coordinate with that
session before editing those files; this document proposes, it does not touch
them.

## The goal, restated

Feed audio one 40 ms frame at a time, update bounded internal state
autoregressively, and emit one motion frame per audio frame, accepting reduced
quality during a cold-start window. This removes the chunk latency floor
entirely: the 4 s -> 1 s retrain reduced the floor; this eliminates it, leaving
frame duration + compute + any deliberate lookahead.

Why no existing model can be pushed there: both ARTalk and its streaming
successor tokenize a whole chunk jointly through a multi-scale codec, so the
coarsest token depends on audio after the frame being emitted. The chunk is a
property of the codec, not the decoder schedule. Frame-level output requires a
frame-level architecture, which is what the scaffold builds.

## What the scaffold already gets right

- **Strict causality by construction**: audio is explicitly framed to 640
  samples before encoding, so training cannot leak future samples.
- **Bounded state**: a 2-layer GRU (constant per-frame cost, no KV growth),
  carrying hidden state + previous emitted frame + a fixed style vector.
- **Style conditioning with dropout** (a null-style vector, CFG-style).
- **A loss set aimed at the right failure**: velocity, acceleration and
  first-frame boundary terms alongside expression/pose/mesh/lips — the terms
  that resist regression-to-the-mean; validation adds LVE/MHD/FDD plus
  velocity/acceleration/jerk statistics.
- **A latency gate**: `check_frame_model.py` fails when p95 per-frame model
  latency exceeds the 40 ms budget.
- **Shared inference package** (`artalk_frame`) imported by both trainer and
  runtime loader, so checkpoints load without a second architecture.

## Gaps against the design space, in priority order

1. **Audio context is 40 ms.** The per-frame conv encoder sees only the
   current frame; phoneme identity typically needs ~100-300 ms. The GRU must
   integrate everything older, through a bottleneck that also carries motion
   dynamics. Cheap, likely high-value fix: encode a sliding window of the last
   k audio frames (still causal), e.g. 8 frames = 320 ms, instead of 1.
2. **No lookahead knob.** Lips anticipate phonemes; strict zero lookahead
   blurs consonant closures. Emitting frame t after seeing audio through
   t+k (k = 2-5 frames, 80-200 ms) is a standard streaming trade and should
   be a config axis, not a constant. Cold-start handling already tolerates
   the added delay.
3. **Data layout: build 108-D data rather than porting the scaffold to
   106.** The scaffold's 108-D contract matches the author's newer,
   recommended layout (his answer: train_code post-dates the released
   model by a year and should be the reference). The source pkls carry
   real eye motion that 106 drops, so the right move is a second LMDB
   built with the builder's `--layout 108` — same split hashing, so the
   held-out identities stay identical across layouts — rather than
   stripping the scaffold's eye dims. The 106 dataset remains for the
   chunk models.
4. **Deterministic regression head.** The known failure mode is muted,
   over-smoothed motion. The velocity/jerk metrics will detect it; the spec
   below stages the escalation (velocity-weighted losses -> lookahead ->
   per-frame discrete tokens with sampling) rather than building the discrete
   head up front.
5. **Fixed teacher forcing (0.75).** Exposure bias at inference argues for a
   decaying schedule (1.0 -> ~0.3 over training); one config knob.
6. **Pipeline surface mismatch.** `FrameByFrameStreamer` lacks the
   `patch_audio_length` / `frames_per_chunk` / `_audio_buffer` attributes the
   realtime pipeline reads. A thin adapter in the app (the artalk1s pattern)
   closes it; with frames_per_chunk=1 the pipeline's chunk machinery
   degenerates gracefully, but the silence pump's chunk-sized budget deserves
   a re-check at that setting.

## Experiment plan

Reuse wholesale: the 106-D LMDB and frozen split, the Slurm runbook
(`--time` explicit, `-x zatou`, sbatch), the generation eval gate
(`eval_gen_generation.py` scores any streamer-shaped model; its vel_ratio and
FDD columns are the mean-collapse detectors), and the rendered side-by-side
tooling.

- **Phase 0 — port and baseline (cheap).** 106-D port of the stream wrapper;
  train the scaffold as-is (GRU 384, 40 ms context, no lookahead). This
  measures how far the minimal architecture gets and validates the loop.
- **Phase 1 — context and lookahead ablation.** Audio window 1/4/8 frames x
  lookahead 0/2/5 on the trained recipe; pick by LVE + vel_ratio + FDD with
  rendered comparisons. Each run is a fraction of the chunk models' cost at
  this model size (~2M params vs 30M).
- **Phase 2 — only if motion is muted:** per-frame residual-quantized tokens
  + sampling head (this is where the design converges with the streaming
  successor's, and where coordinating with its author matters most).
- **Acceptance**: generation gate within an agreed factor of the 1 s model on
  LVE/MHD, vel_ratio >= 0.9, p95 model latency < 40 ms, plus eyes on the
  side-by-sides. Then the app adapter and a live A/B, where the 1 s model is
  the fallback default.

Cold start needs no special machinery: the app's silence pump already primes
the pipeline from connection, so the state warms before the first utterance.
