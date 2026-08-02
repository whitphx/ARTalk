# Chunk-Size Retraining: 4 s → 1 s Latency Floor — Experiment Spec

Status: proposal (coordinate with the streaming-model author before
starting — this experiment is the low-risk complement to the causal
streaming model in preparation, not a replacement for it).

## Goal and hypothesis

The realtime app's dominant remaining latency is the model's 4-second
chunk: ARTalk generates one 100-frame patch per step, so no output exists
until 4 s of audio has accumulated. This is a **training-time
configuration, not an architectural property** — the chunk length is
`CLIP_LENGTH` / `V_PATCH_NUMS[-1]` in the training configs, and the
runtime streamer derives everything from `model.patch_nums`, so a model
trained with a shorter chunk drops into the existing pipeline with zero
code changes.

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

## Open items to resolve with the author first

1. **Motion dimension mismatch**: `train_code` configs say
   `MOTION_DIM: 108` (`100 + 3 + 1 + 2 + 2`), but the released
   checkpoints and the delivered data manifests use 106
   (`exp100 + pose6`). The released models were evidently trained from a
   different config revision. We need the 106-dim config (or the mapping)
   before data prep.
2. **Scale ladder provenance**: release config ladder is
   `[1, 5, 25, 50, 100]`, train_code default `[1, 10, 20, 50, 100]` —
   confirm which matched the released checkpoint.
3. Whether the in-preparation streaming model makes this experiment
   redundant (if it ships soon with frame-level latency, we skip this)
   or complementary (this de-risks the timeline and gives an
   intermediate product).

## Data preparation (from the delivered manifests)

Source: `artalk_processed_data_paths` manifests (TH1KH + CelebV-Text),
50,326 clips / 190.6 h, already in `motion106` format with 16 kHz audio.
Per its README:

1. Combine `trainval` + `test228` + `test_sameid_reference` rows;
   deduplicate by `sample_id`.
2. **Freeze a new held-out split before training** (e.g. 2% val / 2%
   test, stratified by dataset; identities must not straddle splits —
   use the same-identity metadata to enforce this).
3. Build the LMDB with the existing `train_code/core/libs/utils_lmdb.py`
   conventions: per-clip `np.savez` payloads
   `{"audio": float32[N], "motioncode": float32[T, D]}` keyed by
   `lmdb_key`, plus `metadata.json`
   (`{"train": [[key, length], ...], ...}`).
4. Compute `metadata_stats.json` normalization from the new training
   subset only.

Estimated LMDB size: ~10–15 GB (audio + motion only).

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

Integration acceptance (all existing tooling, zero new code):

- `scripts/check_streaming_parity.py --checkpoint <new.pt>` — streaming
  vs one-shot bit-parity at the new chunk size.
- App-repo `scripts/benchmark_pipeline.py --artalk-checkpoint <new.pt>`
  — throughput/VRAM; streamer feed cost is expected to *drop* per chunk
  (shorter encoder input) but run 4x as often.
- Live session: the diagnostics "Turn first frame" metric is the
  headline number — expected to fall from ~1.4–1.7 s toward the sub-1 s
  range (chunk fill at burst delivery + prebuffer + render).

## Timeline sketch

- Week 1: resolve open items; LMDB build + frozen split + stats.
- Weeks 2–3: stage 1 codec at 25 frames; codec eval gate.
- Weeks 3–4: stage 2 generator (+ PREV_LENGTH ablation); offline eval.
- Week 5: integration runbook + live A/B in the app; go/no-go on
  shipping the 1 s model as the app default while the streaming model
  matures.
