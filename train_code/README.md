# ARTalk Training Code

## Data Organization

Training data consists of two parts:

**LMDB database** (`data_lmdb/`): Stores per-clip data as numpy arrays. Each entry is keyed by a clip identifier and contains:
- `audio`: 1D float array, raw waveform at 16kHz sample rate.
- `motioncode`: 2D float array with shape `(num_frames, dim)`, FLAME-based motion parameters at 25fps.

**JSON metadata** (`metadata.json`): Defines train/val/test splits. Format:
```json
{
  "train": [["clip_key_1", seq_length], ["clip_key_2", seq_length], ...],
  "val":   [["clip_key_1", seq_length], ...],
  "test":  [["clip_key_1", seq_length], ...]
}
```

**JSON stats** (`metadata_stats.json`): Normalization statistics for motion parameters, used by the codec model.

Update `DATA_PATH` and `META_PATH` in the config files (`configs/`) to point to your data.

`scripts/build_artalk_lmdb.py` builds this layout from JSONL path manifests (see its docstring for the row schema).

## Training

Training has two stages:

```bash
# Stage 1: Train motion codec (VAE)
accelerate launch train.py -c artalk_codec

# Stage 2: Train AR generation model
# Set VAE_PATH in configs/artalk_gen.yaml to the Stage 1 checkpoint first
accelerate launch train.py -c artalk_gen
```

Optional arguments:
- `--debug`: Debug mode (dry run, no files written or saved).

The `artalk_codec_1s` / `artalk_gen_1s` configs are the same recipe at a 25-frame chunk; see `docs/chunk-size-retraining.md` at the repo root.

## Evaluation

```bash
accelerate launch eval.py -r <checkpoint_path>
```

`scripts/eval_codec_reconstruction.py` and `scripts/eval_gen_generation.py` additionally score a retrained codec or generator against the released checkpoints on the held-out split, and `scripts/render_gen_comparison.py` renders side-by-side comparison videos.
