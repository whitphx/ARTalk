#!/usr/bin/env python
"""Render side-by-side videos: ground truth | retrained | release.

For a few held-out clips, generates motion from audio with both
generators (through their streaming adapters, seeded) and renders the
three motion tracks as one wide mesh video with the original audio, so
generation quality can be judged by eye; the GT-matching metrics cannot
distinguish worse motion from plausibly different motion.
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
from core.libs.utils_lmdb import LMDBEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--release-assets", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from streaming_adapter import ARTalk1sStreamer, load_artalk1s_model
    from artalk.assets import ARTalkAssets
    from artalk.flame_model import RenderMesh
    from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig
    from artalk.streaming import ARTalkStreamer
    from artalk.utils_videos import write_video

    model = load_artalk1s_model(args.checkpoint, args.device)
    new_streamer = ARTalk1sStreamer(model)
    # init_submodule=False codecs carry no FLAME; build one for rendering.
    from core.libs.flame_model import FLAMEModel

    flame = FLAMEModel(n_shape=300, n_exp=100).eval().to(args.device)
    rt = ARTalkRuntime(ARTalkRuntimeConfig(
        assets=ARTalkAssets.resolve(root=str(args.release_assets)),
        device=args.device, flame_scale=1.0,
    ))
    release_streamer = ARTalkStreamer(rt.model)
    mesh = RenderMesh(image_size=args.res, faces=rt.flame_model.get_faces(), scale=1.0)

    def render_track(motion):
        frames = []
        for s in range(0, motion.shape[0], 50):
            batch = motion[s : s + 50].to(args.device)
            verts = flame.get_flame_verts(batch[None], with_headpose=False)[0]
            rgb = mesh(verts)[0]  # (T, 3, H, W) in 0..255
            frames.append(rgb.to(torch.uint8).cpu())
        return torch.cat(frames, dim=0)

    with open(args.data / "metadata.json") as fh:
        meta = json.load(fh)["test"]
    engine = LMDBEngine(str(args.data / "data_lmdb"), write=False)
    args.out.mkdir(parents=True, exist_ok=True)

    done = 0
    for key, length in meta:
        if not 200 <= length <= 400:  # short enough to render quickly, long enough to judge
            continue
        rec = engine[key]
        frames = (length // 100) * 100
        gt = torch.from_numpy(rec["motioncode"]).float()[:frames]
        # Stored audio can run a few frames shorter than the motion track;
        # pad to the frame grid exactly as the training loader does.
        audio = torch.from_numpy(rec["audio"]).float()
        audio = torch.nn.functional.pad(audio, (0, max(0, frames * 640 - audio.shape[0])))[: frames * 640]

        torch.manual_seed(0)
        new_streamer.reset()
        new = new_streamer.feed(audio.to(args.device)).cpu()
        torch.manual_seed(0)
        release_streamer.reset()
        release = release_streamer.feed(audio.to(args.device)).float().cpu()

        panels = [render_track(m) for m in (gt, new, release)]
        video = torch.cat(panels, dim=-1)  # side by side on width
        name = key.replace(":", "_").replace("/", "_")
        out_path = args.out / f"{name}.mp4"
        write_video(video, str(out_path), fps=25,
                    audio_samples=audio.numpy(), sample_rate=16000)
        print(f"wrote {out_path}  (GT | new | release)", flush=True)
        done += 1
        if done >= args.clips:
            break
    engine.close()
    print("DONE")


if __name__ == "__main__":
    main()
