# Chunk-Size Retraining: a 1 s Latency Floor

In a realtime session, ARTalk's dominant latency is the model's 4-second chunk: it generates one 100-frame patch per step, so no motion exists until 4 s of audio has accumulated. That chunk length is a training-time configuration, not an architectural property (`CLIP_LENGTH` / `V_PATCH_NUMS[-1]` in the training configs). This experiment retrains both `train_code` stages at a 25-frame chunk, with more previous-context to compensate for the shorter window, and measures what that costs in motion quality.

One point worth knowing up front: `train_code` does not produce checkpoints in the released model's format. Its generator is a different architecture (separate self-attention plus audio cross-attention per block, codec embedded as `base_codec`) from the released AdaLN VAR-style runtime model; the two have zero state-dict overlap and are not convertible. Serving a retrained checkpoint therefore goes through `scripts/streaming_adapter.py`, which loads the model through the training code itself and exposes the feed/finish/reset surface of `artalk.streaming.ARTalkStreamer`.

## Recipe

`train_code/configs/artalk_codec_1s.yaml` and `artalk_gen_1s.yaml` are the recipe of the released checkpoint with the chunk-size knobs turned, plus `MOTION_DIM` 108 to 106 to match the released checkpoints and the data below. (The in-repo `artalk_codec.yaml` differs from the released checkpoint on two values: the checkpoint's `lvl_idx` buffer encodes the ladder `[1, 5, 25, 50, 100]`, and its `out_mapping` and `motion_mean`/`motion_std` are 106-dim.)

| knob | released checkpoint | 1 s experiment |
| --- | --- | --- |
| `CLIP_LENGTH` | 100 | 25 |
| `V_PATCH_NUMS` | `[1, 5, 25, 50, 100]` | `[1, 5, 25]` |
| `PREV_LENGTH` | 100 (1 chunk) | 75 (3 chunks) |
| `MOTION_DIM` | 106 | 106 |
| everything else | | unchanged |

The two judgment calls:

- **Ladder `[1, 5, 25]`**: the released ladder's own prefix, so the scale ratios (1:5:25) and per-scale token budget match the recipe known to work. The decoder sequence shrinks from 181 to 31 tokens.
- **`PREV_LENGTH` 75**: the previous-context tensor is consumed as whole chunks (`get_motion_feat` splits it by the last patch scale), so 75 frames is three 25-frame chunks, restoring 3 s of the temporal context the shorter window loses. The dataloader zero-pads at clip starts, so early windows train with partial context exactly as the 4 s recipe did. Ablation order if quality disappoints: 75, then 50, then 25. At inference this window is maintained by the streaming adapter; the positional buffers are architecture-derived and train from scratch.

Audio conditioning needs no change: the encoder output is `F.interpolate`d to each patch scale, which works for any chunk length. Style conditioning keeps `STYLE_LENGTH` 100; the style encoder is chunk-length-independent, and the 100-frame style window is why the loader's minimum clip length stays above 100 frames.

Supporting changes to existing `train_code` files, each needed by (but independent of) the 1 s recipe: `get_flame_verts` accepts the 106-dim layout alongside 108, the data loader slices raw motion codes only when they are wider than the model dimension, and `ARTalkGen.inference` maintains its previous-context window at the given length across chunks with the CFG unconditional branch zeroed, which any model trained with `PREV_LENGTH` above one chunk needs.

## Data

`train_code/scripts/build_artalk_lmdb.py` turns path manifests (TH1KH and CelebV processed clips) into the training layout in one pass: the LMDB, the split lists, normalization stats from the training subset only, and a `split_report.json` paper trail. motion106 is `concat(expcode[0:100], posecode[0:6])` per frame.

The split is grouped by source-video identity so no speaker straddles train and held-out: TH1KH groups on the id before `_full_`, CelebV on the id with the two trailing clip indices stripped. Groups are assigned by hashing the identity key (2% val / 2% test), which keeps membership stable if manifests are re-exported. Clips under 103 frames are dropped (one frame above the loader's own minimum, which the 100-frame style window drives), and pairs whose audio and motion lengths disagree by more than five frames are rejected.

The build used here: 47,429 train / 1,128 val / 896 test clips (180.4 h / 4.0 h / 3.2 h).

Two properties of this data to know before comparing against the released model:

- The global head rotation (dims 100 to 102) is exactly zero in every clip; the source processing folds head motion into the per-frame tracking transform. The retrained model therefore holds the head still where the released model sways, and head-motion terms drop out of the losses and the evaluation.
- The released model is out-of-domain on this held-out split, while the retrained model is in-domain. Reconstruction and generation numbers below carry that asymmetry.

## Training

Existing recipes as-is (Adam 1e-4, EMA, 200k iterations), batch sizes unchanged for comparability. Stage 1 (codec) ran on one A100 and was cut short at 150k iterations by a scheduler limit, after its eval gate had already passed. Stage 2 (generator, `VAE_PATH` at the stage-1 checkpoint) ran the full 200k in under 19 h on one A100.

## Results

Evaluation scripts, both in `train_code/scripts/`: `eval_codec_reconstruction.py` (codec round-trip vs the released codec on identical frames) and `eval_gen_generation.py` (generated motion from audio vs ground truth, both models through their streaming paths, seeded). Both skip clips under 200 frames, which leaves 485 of the 896 test clips. `render_gen_comparison.py` renders ground truth, retrained, and released motion side by side with audio for judging by eye.

Stage-1 codec reconstruction (485 clips, 1 s codec vs released codec): LVE 0.493 vs 0.662 mm, mean vertex distance 0.117 vs 0.146 mm, FDD 1.32 vs 1.59, velocity ratio 0.962 vs 0.933. Better or parity on every metric, with the domain caveat above.

Stage-2 generation (same 485 clips, generated motion scored against ground truth):

| metric | 1 s model | released 4 s |
| --- | --- | --- |
| LVE (mm) | 8.76 | 7.85 |
| mean vertex distance (mm) | 1.97 | 1.77 |
| FDD | 23.30 | 33.42 |
| velocity ratio vs GT | 0.936 | 0.609 |

The 1 s model is roughly 12% behind on the ground-truth-frame-matching metrics and 30% ahead on upper-face dynamics, and it preserves nearly all of the ground truth's motion energy where the released model keeps 61%. The two models trade frame accuracy against liveliness, and the frame-matching metrics punish lively-but-different motion, so rendered videos are the tiebreaker.

Live, in a realtime avatar application built on this repo's `artalk` runtime, the 1 s model's lip-sync holds up well and the first motion of a turn arrives correspondingly earlier. The visible regression is eye motion: the eye region reads noticeably less natural than the released model's. motion106 carries no eye-pose channels, so everything the eyes do lives in the expression coefficients, where the two models learned different behavior.

## Open questions

- **Eye motion.** Whether the eye-region regression traces to the data (the zero head rotation hints at aggressive normalization elsewhere too), the shorter chunk, or the recipe knobs is open; it is the main quality question before shipping a 1 s model.
- **Deployed chunk composition.** In streaming use, every turn ends on a partial chunk zero-padded to full length, and every turn begins from silent previous context. The second case measured benign (a speech chunk decoded after a silence chunk shows the same movement energy as one decoded fresh, mean frame difference 0.374 vs 0.350); the first is untested. Drawing some training windows with zero-padded tails would cover it without new data.
- **Style conditioning** is trained but untested here; all evaluation ran with the zero (unconditional) style input.
