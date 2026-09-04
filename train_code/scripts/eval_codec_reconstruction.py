#!/usr/bin/env python
"""Codec eval gate: reconstruction quality of a retrained codec vs the release.

Round-trips the held-out test split through both codecs over identical
frames and reports the standard talking-head metrics computed by
``core.models.modules.metrics.calc_val_metrics``:

  LVE  lip vertex error (mm): per-frame max lip-vertex distance, averaged
  MHD  mean vertex distance (mm) over the whole head
  FDD  upper-face dynamics deviation (x100)

plus expression/jaw L1 in code space and a motion-energy ratio
(reconstructed frame-to-frame vertex velocity over ground truth; 1.0
means the codec neither smooths away nor invents motion).

The release codec reconstructs 100-frame chunk pairs, so only clips with
at least 200 frames enter the comparison, trimmed to a multiple of 100;
the new codec sees exactly the same frames in its own window size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The repo root provides the released `artalk` package without an install.
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
from core.libs.utils import ConfigDict  # noqa: E402
from core.libs.utils_lmdb import LMDBEngine  # noqa: E402
from core.models import build_model  # noqa: E402
from core.models.modules.metrics import calc_val_metrics  # noqa: E402


def reconstruct_new(codec, motion, window):
    wins = motion.unfold(0, window, window).permute(0, 2, 1)  # [N, window, C]
    bits = codec.quant_to_vqidx(wins)
    return codec.vqidx_to_motion(bits).reshape(-1, motion.shape[-1])


def reconstruct_release(vae, motion, window=100):
    wins = [motion[i : i + window] for i in range(0, motion.shape[0], window)]
    out = [None] * len(wins)
    for i in range(len(wins) - 1):
        prev_idx, this_idx = vae.quant_to_vqidx(wins[i][None], wins[i + 1][None])
        prev_rec, this_rec = vae.vqidx_to_motion(prev_idx, this_idx)
        if i == 0:
            out[0] = prev_rec[0]
        out[i + 1] = this_rec[0]
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path, help="retrained codec .pt")
    ap.add_argument("--data", required=True, type=Path, help="dataset dir (data_lmdb + metadata.json)")
    ap.add_argument("--release-assets", type=Path, default=None,
                    help="ARTalk asset root; enables the released-codec baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta = ConfigDict(ck["meta_cfg"], gpus=1, cli_args=[])
    codec = build_model(meta.MODEL)
    # The trainer's EMA snapshot omits the FLAME constants, which come
    # from the asset file at construction; nothing else may be absent.
    missing, unexpected = codec.load_state_dict(ck["model"], strict=False)
    stray = [k for k in missing if not k.startswith("face_decoder.")]
    if stray or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={stray}, unexpected={unexpected}")
    codec.eval().to(args.device)
    window = codec.patch_nums[-1]
    flame = codec.face_decoder.to(args.device)

    release_vae = None
    if args.release_assets is not None:
        from artalk.assets import ARTalkAssets
        from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig

        rt = ARTalkRuntime(ARTalkRuntimeConfig(
            assets=ARTalkAssets.resolve(root=str(args.release_assets)),
            device=args.device, flame_scale=1.0,
        ))
        release_vae = rt.model.basic_vae

    with open(args.data / "metadata.json") as fh:
        meta_json = json.load(fh)[args.split]
    engine = LMDBEngine(str(args.data / "data_lmdb"), write=False)

    def flame_verts(m):
        vs = []
        for s in range(0, m.shape[0], 200):
            vs.append(flame.get_flame_verts(m[None, s : s + 200], with_headpose=False)[0])
        return torch.cat(vs, dim=0)

    sums: dict[str, list] = {}
    n_clips = 0
    for key, length in meta_json:
        if length < 200:
            continue
        motion = torch.from_numpy(engine[key]["motioncode"]).float().to(args.device)
        motion = motion[: (motion.shape[0] // 100) * 100]
        gt_verts = flame_verts(motion)
        gt_vel = (gt_verts[1:] - gt_verts[:-1]).norm(dim=-1).mean()

        recons = {"new": reconstruct_new(codec, motion, window)}
        if release_vae is not None:
            recons["release"] = reconstruct_release(release_vae, motion)
        for name, rec in recons.items():
            verts = flame_verts(rec)
            lve, mhd, fdd = calc_val_metrics(verts[None], gt_verts[None])
            vel = (verts[1:] - verts[:-1]).norm(dim=-1).mean()
            row = {
                "exp_l1": (rec[:, :100] - motion[:, :100]).abs().mean().item(),
                "jaw_l1": (rec[:, 103:106] - motion[:, 103:106]).abs().mean().item(),
                "lve_mm": lve.item(),
                "mhd_mm": mhd.item(),
                "fdd": fdd.item(),
                "vel_ratio": (vel / gt_vel).item(),
            }
            sums.setdefault(name, []).append(row)
        n_clips += 1
        if args.max_clips and n_clips >= args.max_clips:
            break
    engine.close()

    print(f"split={args.split}  clips evaluated={n_clips} (>=200 frames each)")
    header = ["codec", "exp_l1", "jaw_l1", "lve_mm", "mhd_mm", "fdd", "vel_ratio"]
    print(" ".join(f"{h:>9}" for h in header))
    for name, rows in sums.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{name:>9} " + " ".join(f"{m[k]:9.4f}" for k in header[1:]))


if __name__ == "__main__":
    main()
