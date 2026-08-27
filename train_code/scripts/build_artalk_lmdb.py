#!/usr/bin/env python
"""Build the ARTalk training LMDB from processed-data path manifests.

Input: JSONL manifests whose rows point at a 16 kHz mono wav and a
``smoothed.pkl`` of per-frame FLAME codes. Output, under ``--out``:

  data_lmdb/            np.savez payloads {"audio": [N], "motioncode": [T, 106]}
  metadata.json         {"train"|"val"|"test": [[lmdb_key, n_frames], ...]}
  metadata_stats.json   {"motion_mean": [106], "motion_std": [106]} from train only
  split_report.json     what went where and why, for the paper trail

motion106 is ``concat(expcode[0:100], posecode[0:6])`` per frame, matching
the delivered manifests' README and the released checkpoints (the training
configs' 108-dim layout belongs to a different data revision).

The split is grouped by identity so no speaker straddles train and
held-out: clips sharing a source video id form one group, and groups are
assigned to splits by hash of the identity key, which keeps membership
stable when manifests are re-exported or rows are added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.libs.utils_lmdb import LMDBEngine  # noqa: E402

import torch
import torchaudio

MOTION_DIM = 106
SAMPLE_RATE = 16_000
FPS = 25
# The dataset loader keeps only clips longer than max(100, CLIP_LENGTH) + 1
# frames (the style window needs 100 frames), so anything at or below this
# is dead weight in the database.
MIN_FRAMES = 102


def identity_key(dataset: str, sample_id: str) -> str:
    """Source-video identity for split grouping.

    th1kh ids look like ``<video>_full_<clip>_S..``; celebv ids look like
    ``<video>_<clip>_<seg>`` where <video> is a YouTube id that may itself
    contain underscores, so strip the two trailing numeric tokens.
    """
    if dataset == "th1kh":
        return f"th1kh:{sample_id.split('_full_')[0]}"
    parts = sample_id.rsplit("_", 2)
    return f"{dataset}:{parts[0] if len(parts) == 3 else sample_id}"


def assign_split(identity: str, val_pct: float, test_pct: float) -> str:
    # Hash-based assignment: deterministic, order-independent, and stable
    # under re-exports that add or drop rows.
    h = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    u = h / 2**64
    if u < test_pct:
        return "test"
    if u < test_pct + val_pct:
        return "val"
    return "train"


def load_motion106(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as fh:
        frames = pickle.load(fh)
    # Frame keys end in _<index>; numeric order, not lexicographic.
    keys = sorted(frames.keys(), key=lambda k: int(k.rsplit("_", 1)[1]))
    out = np.empty((len(keys), MOTION_DIM), dtype=np.float32)
    for i, key in enumerate(keys):
        f = frames[key]
        out[i, :100] = f["expcode"]
        out[i, 100:] = f["posecode"]
    return out


def load_audio(path: str) -> np.ndarray:
    # Sources are stored at various rates (44.1 kHz for TH1KH); the model
    # consumes 16 kHz mono.
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
    return np.ascontiguousarray(wav.numpy(), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-pct", default=0.02, type=float)
    ap.add_argument("--test-pct", default=0.02, type=float)
    ap.add_argument("--limit", type=int, default=None, help="First N rows only (smoke runs).")
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for m in args.manifests:
        for line in open(m):
            r = json.loads(line)
            rows.setdefault(r["sample_id"], r)  # dedupe across manifests
    items = list(rows.values())[: args.limit]
    print(f"{len(items)} unique clips from {len(args.manifests)} manifests")

    args.out.mkdir(parents=True, exist_ok=True)
    engine = LMDBEngine(str(args.out / "data_lmdb"), write=True)
    metadata = defaultdict(list)
    split_report = defaultdict(lambda: defaultdict(int))
    stats_count = 0
    stats_sum = np.zeros(MOTION_DIM, dtype=np.float64)
    stats_sqsum = np.zeros(MOTION_DIM, dtype=np.float64)
    skipped = defaultdict(int)

    for i, r in enumerate(items):
        identity = identity_key(r["dataset"], r["sample_id"])
        split = assign_split(identity, args.val_pct, args.test_pct)
        try:
            motion = load_motion106(r["gt_smoothed_pkl_path"])
            audio = load_audio(r["audio_path"])
        except (OSError, ValueError, KeyError) as exc:
            skipped[f"load error ({type(exc).__name__})"] += 1
            continue
        if motion.shape[0] <= MIN_FRAMES:
            skipped["too short"] += 1
            continue
        # Trust the shorter of the two streams; misalignment beyond one
        # frame means the pair is suspect.
        expected = int(motion.shape[0] * SAMPLE_RATE / FPS)
        if abs(audio.shape[0] - expected) > 5 * SAMPLE_RATE / FPS:
            skipped["audio/motion length mismatch"] += 1
            continue
        key = f"{r['dataset']}:{r['sample_id']}"
        engine.dump(key, {"audio": audio, "motioncode": motion})
        metadata[split].append([key, int(motion.shape[0])])
        split_report[split][r["dataset"]] += 1
        if split == "train":
            stats_count += motion.shape[0]
            stats_sum += motion.sum(axis=0)
            stats_sqsum += (motion.astype(np.float64) ** 2).sum(axis=0)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    engine.close()

    mean = stats_sum / stats_count
    std = np.sqrt(np.maximum(stats_sqsum / stats_count - mean**2, 1e-12))
    (args.out / "metadata.json").write_text(json.dumps(dict(metadata)))
    (args.out / "metadata_stats.json").write_text(
        json.dumps({"motion_mean": mean.tolist(), "motion_std": std.tolist()})
    )
    (args.out / "split_report.json").write_text(
        json.dumps(
            {
                "splits": {k: dict(v) for k, v in split_report.items()},
                "clips": {k: len(v) for k, v in metadata.items()},
                "hours": {
                    k: round(sum(n for _, n in v) / FPS / 3600, 2)
                    for k, v in metadata.items()
                },
                "skipped": dict(skipped),
                "val_pct": args.val_pct,
                "test_pct": args.test_pct,
            },
            indent=1,
        )
    )
    print("splits:", {k: len(v) for k, v in metadata.items()})
    print("skipped:", dict(skipped))
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
