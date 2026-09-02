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

**Stage A — per-frame codec.** `ARTalkCodec` with `DATASET.CLIP_LENGTH 1`
and `MODEL.V_PATCH_NUMS [1]`: one 32-bit BSQ token per frame, no
temporal mixing, so it is causal by construction and needs no mask.
Verified to build and round-trip `(B, 1, 108) -> bits (B, 1, 32) ->
(B, 1, 108)`. Losses and evaluation are the codec gate unchanged
(`eval_codec_reconstruction.py` with `window = 1`). The bit budget is
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
conditioning, the streamer surface — is Phase 0 code. The frame adapter
in the app needs one branch: decode tokens through the codec before
returning motion.

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
velocity ratio 0.470 / 0.469 at 50k / 80k iterations; the full-split
endpoint numbers follow when the eval job completes. Its checkpoint
remains useful as the control row for Phase 2 and as a latency baseline.
