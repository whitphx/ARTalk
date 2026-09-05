# Frame-Level Generation with Sampled Tokens (Phase 2)

Status: experiment spec. Follows `frame-streaming-assessment.md`; the
Phase 0 baseline it escalates from is described at the end.

## Why a sampling head, not a better regressor

The Phase 0 frame model (per-frame audio encoder, GRU state, direct
regression to motion) trains well but generates muted motion: its
generated-motion velocity ratio sat at 0.47 of ground truth from 25% of
training onward, flat while every teacher-forced loss kept improving. A
deterministic head trained on L1/L2 converges to the conditional mean of
plausible motions, which is smooth and small. Wider audio context and
lookahead (Phase 1) change *what* the lips say, not how quietly; they
stack on top of this design rather than replace it.

The chunk models avoid the collapse because they emit discrete codec
tokens and *sample* them. The retrained 1 s generator's velocity ratio is
0.83-0.96 for that reason. Phase 2 brings the same mechanism to frame
level: a per-frame codec and a causal generator that samples one token
per frame.

## Design

Two stages, mirroring the chunk recipe, both built from pieces that
already exist in `train_code`.

**Stage A — per-frame codec.** `ARTalkCodec` with `V_PATCH_NUMS [1]`
and `FRAME_INDEPENDENT: True` (`configs/artalk_codec_frame.yaml`): every
frame is encoded, quantized to one 32-bit BSQ token and decoded on its
own, so the codec is causal by construction and needs no mask. Training
still reads 25-frame windows, which keeps the loader efficient and lets
the velocity and smoothness losses penalize frame-to-frame
reconstruction jitter; a frame reconstructs identically alone or inside
a window (verified). Evaluation is the codec gate unchanged
(`eval_codec_reconstruction.py`, whose window follows `patch_nums`). The bit budget is
32 bits/frame against the 25-frame codec's ~40 bits/frame, so
reconstruction should land near it; if it falls short, the escalation is
two tokens per frame (`V_PATCH_NUMS [1, 1]` is not a valid ladder — use
`V_CODE_DIM 64` instead) before considering a short causal window
(`CLIP_LENGTH k`, `V_PATCH_NUMS [1, k]`, causal `attn_mask`), which
also builds but reintroduces k-1 frames of decode dependence.

**Stage B — causal token generator.** `CausalFrameModel` with its two
regression heads replaced by a bits head: hidden state -> `(32, 2)`
logits per frame, trained with the same per-bit cross-entropy as
`ARTalkGen` and sampled with its `sample_idx_with_top_p_` (top-p 0.97,
temperature knob). The sampled token decodes through the frozen Stage A
codec into the frame that is emitted *and* fed back as previous motion.
Training feeds ground-truth tokens as previous-motion input (standard
teacher forcing on tokens; the scheduled-sampling ratio the scaffold
already has applies unchanged). Classifier-free guidance on audio and
style follows `ARTalkGen`'s `AUDIO_FREE` / `STYLE_FREE` dropout, since
guidance is a second liveliness lever at inference.

Everything else — per-frame audio encoder, GRU state, style
conditioning, the streamer surface — is Phase 0 code. Implemented as
`core/models/artalk_frame_gen` (`configs/artalk_frame_token.yaml`); the
app's frame adapter builds token checkpoints through the training
registry and streams them through the same per-frame step API.

## Per-frame cost

Audio encoder step + GRU step + bits head + codec decode of a single
token: well under the 40 ms budget on any GPU in the fleet, and the
existing latency gate (`check_frame_model.py`) measures it directly.

## Acceptance

The generation gate, same protocol as every model so far:

- `vel_ratio >= 0.9` (Phase 0: 0.47; release 4 s: 0.93; 1 s: 0.96)
- LVE / MHD within an agreed factor of the 1 s chunk model
- FDD in family with the release model
- p95 per-frame model latency < 40 ms
- rendered side-by-sides (`render_gen_comparison.py`), because the
  L1-family metrics cannot tell muted from differently alive

## Stage A result

Both per-frame codecs trained to 200k on the 108-D data. The 64-bit
variant is the codec of record (100 held-out clips, release codec under
the same protocol):

| codec | exp L1 | jaw L1 | LVE mm | MHD mm | FDD | vel_ratio |
|---|---|---|---|---|---|---|
| 32-bit (val slice) | | | 1.93 | 0.55 | 11.0 | |
| 64-bit | 0.093 | 0.015 | 1.14 | 0.331 | 5.86 | 1.08 |
| release | 0.037 | 0.048 | 0.667 | 0.641 | 1.58 | 0.79 |

Whole-head error at half the release's and jaw three times better;
lips at 1.7x, the per-frame budget's remaining cost. The velocity ratio
slightly above one flags mild high-frequency excess in reconstruction
(0.97 at 50k), worth re-checking on generated motion. Most of the
improvement came in the learning-rate decay phase: the 64-bit codec
halved its errors between 100k and 200k, so codec runs should not be
judged before their endpoint.

## Pilot result

A pilot of this design (generator trained 200k iterations on the 64-bit
per-frame codec's 50k checkpoint) on the full held-out split, same
protocol as the baseline table below:

| model | LVE mm | MHD mm | FDD | vel_ratio |
|---|---|---|---|---|
| regression baseline | 10.10 | 2.35 | 40.8 | 0.43 |
| token pilot | 9.98 | 2.29 | 31.2 | 0.898 |
| release (4 s) | 7.86 | 1.89 | 33.4 | 0.52 |

Sampling restores motion energy at frame level (0.43 -> 0.90, at the
acceptance line within noise; 0.914 on the 20-clip sample) and brings
upper-face dynamics below the release's. Lip and mesh error sit at
1.27x the release: the per-frame codec's ~2 mm lip floor (its gate:
LVE 2.11 vs the release codec's 0.67, MHD at parity, vel_ratio 0.97 so
no jitter) plus 40 ms of audio context and zero lookahead. Those are
the Phase 1 levers and the sliding-window codec, in that order.

## Sequencing

1. Stage A on the 108-D dataset (`~/data/artalk-108-data`), codec gate.
   Cheap: the 1 s codec reached release quality in 150k iterations at
   ~2 it/s; per-frame windows are far smaller.
2. Stage B, generation gate. Expect the velocity ratio to move first.
3. Only then Phase 1 knobs (audio context window, lookahead), as an
   ablation on the token model.

Slurm runbook as before: `sbatch`, explicit `--time`, `-x zatou`, the
frozen-snapshot discipline for any code shared with another session.

## The Phase 0 baseline this escalates from

`CausalFrameModel` regression head, 2.64M parameters, 200k iterations at
batch 32 (~1.56 it/s, 32 h on one A100). Generated-motion metrics on 20
held-out clips: LVE 8.08 / 8.16 mm, MHD 1.94 / 1.96 mm, FDD 35.9 / 39.3,
velocity ratio 0.470 / 0.469 at 50k / 80k iterations. Full-split endpoint
(485 clips, 200k), with the release model under the identical protocol:

| model | LVE mm | MHD mm | FDD | vel_ratio |
|---|---|---|---|---|
| frame baseline | 10.10 | 2.35 | 40.8 | 0.43 |
| release (4 s) | 7.86 | 1.89 | 33.4 | 0.52 |

Protocol note: ground truth here is 108-D and carries eye motion, so a
106-D model's velocity ratio reads lower than on 106-D ground truth (the
release measured 0.93 there). The Phase 2 acceptance threshold applies
to 108-D outputs. The baseline checkpoint remains the control row and a
latency baseline.
