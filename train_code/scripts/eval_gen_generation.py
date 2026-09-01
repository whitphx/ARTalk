#!/usr/bin/env python
"""Generator eval gate: generated-motion quality vs the released model.

Generates motion from held-out audio with both generators and scores it
against ground truth via ``calc_val_metrics`` (LVE / MHD / FDD) plus a
velocity ratio. The retrained model runs through its streaming adapter so
context lengths match training and the numbers reflect the app's actual
serving path; the release runs through the packaged ``ARTalkStreamer``
for the same reason. Sampling is seeded per clip for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.libs.utils_lmdb import LMDBEngine  # noqa: E402
from core.models.modules.metrics import calc_val_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--model-kind", choices=("artalk1s", "frame"), default="artalk1s")
    ap.add_argument("--frame-package-dir", type=Path, default=None,
                    help="artalk_frame package location for --model-kind frame")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--app-repo", required=True, type=Path,
                    help="artalk-streamlit-realtime checkout (provides the adapter)")
    ap.add_argument("--release-assets", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(args.app_repo))
    if args.model_kind == "frame":
        from artalk_streamlit_realtime.framemodel import FrameModelStreamer, load_frame_model

        if args.frame_package_dir is None:
            raise SystemExit("--frame-package-dir is required for --model-kind frame")
        model = load_frame_model(args.frame_package_dir, args.checkpoint, args.device)
        # Native layout: a 108-trained model is scored against 108 ground
        # truth with its eye channel intact.
        new_streamer = FrameModelStreamer(model, native_layout=True)
    else:
        from artalk_streamlit_realtime.artalk1s import ARTalk1sStreamer, load_artalk1s_model

        model = load_artalk1s_model(
            Path(__file__).resolve().parent.parent, args.checkpoint, args.device
        )
        new_streamer = ARTalk1sStreamer(model)
    # init_submodule=False codecs carry no FLAME; build one for scoring.
    from core.libs.flame_model import FLAMEModel

    flame = FLAMEModel(n_shape=300, n_exp=100).eval().to(args.device)

    release_streamer = None
    if args.release_assets is not None:
        from artalk.assets import ARTalkAssets
        from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig
        from artalk.streaming import ARTalkStreamer

        rt = ARTalkRuntime(ARTalkRuntimeConfig(
            assets=ARTalkAssets.resolve(root=str(args.release_assets)),
            device=args.device, flame_scale=1.0,
        ))
        release_streamer = ARTalkStreamer(rt.model)

    meta = json.load(open(args.data / "metadata.json"))[args.split]
    engine = LMDBEngine(str(args.data / "data_lmdb"), write=False)

    def verts_of(m):
        vs = []
        for s in range(0, m.shape[0], 200):
            vs.append(flame.get_flame_verts(m[None, s : s + 200].to(args.device), with_headpose=False)[0])
        return torch.cat(vs, dim=0)

    sums: dict[str, list] = {}
    n = 0
    for key, length in meta:
        if length < 200:
            continue
        rec = engine[key]
        frames = (length // 100) * 100
        gt = torch.from_numpy(rec["motioncode"]).float()[:frames]
        # Stored audio can run a few frames shorter than the motion track;
        # pad to the frame grid exactly as the training loader does.
        audio = torch.from_numpy(rec["audio"]).float()
        audio = torch.nn.functional.pad(audio, (0, max(0, frames * 640 - audio.shape[0])))[: frames * 640]
        gt_verts = verts_of(gt)
        gt_vel = (gt_verts[1:] - gt_verts[:-1]).norm(dim=-1).mean()

        gens = {}
        torch.manual_seed(0)
        new_streamer.reset()
        gens["new"] = new_streamer.feed(audio.to(args.device)).cpu()
        if release_streamer is not None:
            torch.manual_seed(0)
            release_streamer.reset()
            gens["release"] = release_streamer.feed(audio.to(args.device)).float().cpu()

        for name, gen in gens.items():
            if gen.shape[0] != frames:
                raise RuntimeError(f"{name}: {gen.shape[0]} frames for {frames} expected")
            gv = verts_of(gen)
            lve, mhd, fdd = calc_val_metrics(gv[None], gt_verts[None])
            vel = (gv[1:] - gv[:-1]).norm(dim=-1).mean()
            sums.setdefault(name, []).append({
                "lve_mm": lve.item(), "mhd_mm": mhd.item(), "fdd": fdd.item(),
                "vel_ratio": (vel / gt_vel).item(),
            })
        n += 1
        if args.max_clips and n >= args.max_clips:
            break
    engine.close()

    print(f"split={args.split}  clips={n} (generation from audio, seeded)")
    header = ["model", "lve_mm", "mhd_mm", "fdd", "vel_ratio"]
    print(" ".join(f"{h:>9}" for h in header))
    for name, rows in sums.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{name:>9} " + " ".join(f"{m[k]:9.4f}" for k in header[1:]))


if __name__ == "__main__":
    main()
