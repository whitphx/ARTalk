# Chunk-Size Retraining: 4 s → 1 s Latency Floor — Experiment Spec

Status: proposal (coordinate with the streaming-model author before
starting — this experiment is the low-risk complement to the causal
streaming model in preparation, not a replacement for it).

## Goal and hypothesis

The realtime app's dominant remaining latency is the model's 4-second
chunk: ARTalk generates one 100-frame patch per step, so no output exists
until 4 s of audio has accumulated. This is a **training-time
configuration, not an architectural property** — the chunk length is
`CLIP_LENGTH` / `V_PATCH_NUMS[-1]` in the training configs.

One earlier assumption did not survive contact with the code: this
`train_code` does **not** produce checkpoints in the released model's
format. Its generator is a newer architecture (separate self-attention
plus audio cross-attention per block, codec embedded as `base_codec`),
sharing its design with the author's streaming successor, while the
released runtime model is the AdaLN VAR style; the two have zero
state-dict overlap and are not convertible. Integration therefore goes
through a thin streamer adapter around the train-code model's own
`inference()` — the same feed/finish pattern already proven for the
successor model, injected via the pipeline's `streamer=` parameter. The
stage-2 checkpoint is self-contained (codec weights included), which the
adapter can exploit exactly as the successor's loader does.

Hypothesis: a 25-frame (1 s) chunk model, given more previous-context
(`PREV_LENGTH`), retains acceptable motion quality while cutting the
latency floor 4x. Fallback midpoint: 50 frames (2 s).

## What changes (both training stages)

The chunk length is baked into both stages, so this is a two-stage
retrain using the existing `train_code/`:

1. **Stage 1 — motion codec** (`configs/artalk_codec.yaml`):
   - `DATASET.CLIP_LENGTH: 100 → 25`
   - `MODEL.V_PATCH_NUMS: [1, 10, 20, 50, 100] → [1, 5, 25]`
     (multi-scale ladder must end at the chunk length; keep roughly
     geometric steps)
2. **Stage 2 — AR generator** (`configs/artalk_gen.yaml`):
   - same `CLIP_LENGTH` / `V_PATCH_NUMS` changes (mirrored in
     `VAE_CONFIG`), `VAE_PATH` pointing at the stage-1 checkpoint
   - `DATASET.PREV_LENGTH: 100 → 75` (three previous chunks — restores
     the temporal context the shorter window loses; ablate 25/50/75)
   - runtime equivalent is `AR_CONFIG.PREV_RATIO` (released config uses
     1 previous chunk; the 1 s variant should ship with 3)

Audio conditioning needs no architecture change: the encoder output is
`F.interpolate`d to each patch scale, which works for any chunk length.

## Formerly open items, now resolved

1. **Motion dimension — resolved: 106.** The released checkpoint is
   106-dim end to end (`basic_vae.decoder.out_mapping` is `(106, 512)`,
   `motion_mean`/`motion_std` are `(106,)`), matching the manifests. The
   108-dim training configs belong to a different data revision whose raw
   `motioncode` is wider and sliced in the loader; the loader now skips
   that slice when the stored code is already model-sized.
2. **Scale ladder — resolved: `[1, 5, 25, 50, 100]`.** Read off the
   released checkpoint's `lvl_idx` buffer (level counts 1/5/25/50/100,
   sum 181 = `pos_embed` length). The 1 s ladder `[1, 5, 25]` is that
   ladder's prefix, so the per-scale ratios match the released recipe.
3. **Streaming-model overlap — complementary, with measurements.** The
   streaming successor's current checkpoint still generates in 4 s
   chunks, and its motion decode alone measures 0.72x realtime on a P100
   (83% of it in 176 sequential decoder forwards), so it does not lower
   the latency floor today. The 1 s retrain attacks the floor directly
   and also shortens that successor's own path: its step count per chunk
   is bounded by frames per chunk, so the same chunk-size lever helps
   both models.

## Data preparation (implemented)

`train_code/scripts/build_artalk_lmdb.py` turns the manifests into the
training layout in one pass: LMDB (`np.savez` payloads
`{"audio": float32[N], "motioncode": float32[T, 106]}`), `metadata.json`
split lists, `metadata_stats.json` from the training subset only, and a
`split_report.json` paper trail. motion106 is built per frame as
`concat(expcode[0:100], posecode[0:6])` from each clip's
`smoothed.pkl`, frames ordered by their numeric suffix.

The split is grouped by source-video identity so no speaker straddles
train and held-out: TH1KH groups on the id before `_full_` (1,189
groups over 27,794 clips), CelebV on the id with the two trailing clip
indices stripped. Groups are assigned by hashing the identity key
(default 2% val / 2% test), which keeps membership stable if manifests
are re-exported. Clips at or under 102 frames are dropped to match the
loader's own minimum (0.1% of rows), and pairs whose audio and motion
lengths disagree by more than five frames are rejected.

One property of this data to know before comparing against the released
model: the global head rotation (dims 100-102) is exactly zero in every
clip — the source processing folds head motion into the per-frame
tracking transform. The retrained model will therefore hold the head
still. The photoreal render path zeroes head pose anyway, so the app is
unaffected; the mesh preview loses the released model's head sway, and
head-motion terms drop out of both the losses and the evaluation.

Smoke-check the pipeline with `--limit 200` before the full build. The
source data is on the shared `/data` mount; the full build is I/O-bound
and safe to run on any host that mounts it.

Estimated LMDB size: ~10-15 GB (audio + motion only).

## Model configuration (implemented)

`train_code/configs/artalk_codec_1s.yaml` and `artalk_gen_1s.yaml` are
the released 4 s recipe with exactly the chunk-size knobs turned:

| knob | released 4 s | 1 s experiment |
| --- | --- | --- |
| `CLIP_LENGTH` | 100 | 25 |
| `V_PATCH_NUMS` | `[1, 5, 25, 50, 100]` | `[1, 5, 25]` |
| `PREV_LENGTH` | 100 (1 chunk) | 75 (3 chunks) |
| `MOTION_DIM` | 106 | 106 |
| everything else | — | unchanged |

Rationale for the two judgment calls:

- **Ladder `[1, 5, 25]`**: the released ladder's own prefix, so scale
  ratios (1:5:25) and the per-scale token budget match the recipe that
  is known to work. The decoder sequence shrinks from 181 to 31 tokens.
- **`PREV_LENGTH` 75**: the previous-context tensor is consumed as
  whole chunks (`get_motion_feat` splits it by the last patch scale), so
  75 frames = three 25-frame chunks. This restores 3 s of the temporal
  context the shorter window loses; the dataloader zero-pads at clip
  starts, so early windows train with partial context exactly as the 4 s
  recipe did. Ablation order if quality disappoints: 75 -> 50 -> 25.
  At runtime this is `prev_ratio` 3 in the model config; the positional
  buffers (`prev_pos_embed` et al.) are architecture-derived and train
  from scratch.

Style conditioning keeps `STYLE_LENGTH` 100: the style encoder is
chunk-length-independent, and 100-frame style windows are why the
loader's minimum clip length stays at 102 frames.

## Training protocol

Existing recipes as-is (Adam 1e-4, EMA, 200k iterations), batch sizes
unchanged (shorter clips reduce memory; batch could grow, but keep the
recipe fixed for comparability). Both stages need a tensor-core GPU;
the A100 host fits. Rough wall-clock at the recipe's iteration counts:
days per stage, not weeks — and Stage 2 supports `PRETRAIN_PATH` if a
warm start from the released generator proves useful (embedding shapes
that depend on patch count will not transfer; expect scratch to be the
honest baseline).

## Evaluation

Offline (against the frozen test split):

- Codec: reconstruction losses per the stage-1 recipe (expression /
  pose / mesh / lips terms already defined in `LOSS_KWARGS`).
- Generator: teacher-forced CE + generated-motion metrics vs ground
  truth (vertex error, lip-region error, velocity distributions), 1 s
  model vs released 4 s model on identical inputs.
- Side-by-side rendered videos for the team (the app's mesh renderer
  suffices).

Integration acceptance:

- New (small): the train-code-architecture streamer adapter described
  above, plus a parity check of chunked feed against the model's own
  one-shot `inference()`.
- App-repo `scripts/benchmark_pipeline.py --artalk-checkpoint <new.pt>`
  — throughput/VRAM; streamer feed cost is expected to *drop* per chunk
  (shorter encoder input) but run 4x as often.
- Live session: the diagnostics "Turn first frame" metric is the
  headline number — expected to fall from ~1.4–1.7 s toward the sub-1 s
  range (chunk fill at burst delivery + prebuffer + render).

## Timeline sketch

- Week 1: LMDB build + frozen split + stats (open items are resolved;
  the builder script exists — this is now a run, not a design task).
- Weeks 2–3: stage 1 codec at 25 frames; codec eval gate.
- Weeks 3–4: stage 2 generator (+ PREV_LENGTH ablation); offline eval.
- Week 5: integration runbook + live A/B in the app; go/no-go on
  shipping the 1 s model as the app default while the streaming model
  matures.
