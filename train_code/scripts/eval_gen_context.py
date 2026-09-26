#!/usr/bin/env python
"""A/B two 1 s generators through the model's own ``inference()``.

The standard gate drives the retrained model through the app's streaming
adapter, which hands the generator one chunk at a time and so gives a
longer-audio-context model no context to attend to. ``inference()``
carries that context across chunks itself, so running both arms through
it scores them on the same footing, with the audio context the context
model was trained with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.libs.flame_model import FLAMEModel  # noqa: E402
from core.libs.utils import ConfigDict  # noqa: E402
from core.libs.utils_lmdb import LMDBEngine  # noqa: E402
from core.models import build_model  # noqa: E402
from core.models.modules.metrics import calc_val_metrics  # noqa: E402


def load(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ConfigDict(ckpt["meta_cfg"], gpus=1, cli_args=[])
    model = build_model(cfg.MODEL, init_submodule=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    stray = [k for k in missing if not k.startswith(("audio_encoder.", "base_codec.face_decoder."))]
    if stray or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={stray}, unexpected={unexpected}")
    return model.eval().to(device), getattr(model, "audio_context_patches", 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path, action="append",
                    help="repeatable; each is scored as its own row")
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--force-context", action="append", type=int, default=None,
                    help="repeatable, per checkpoint: override how many past chunks it "
                         "attends to. Scoring a context model at 0 says whether the "
                         "context is doing anything, rather than merely being supplied.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    labels = args.label or [c.parent.parent.name for c in args.checkpoint]
    flame = FLAMEModel(n_shape=300, n_exp=100).eval().to(args.device)

    def verts_of(m):
        vs = []
        for s in range(0, m.shape[0], 200):
            vs.append(flame.get_flame_verts(m[None, s:s + 200].to(args.device), with_headpose=False)[0])
        return torch.cat(vs, dim=0)

    meta = json.load(open(args.data / "metadata.json"))[args.split]
    engine = LMDBEngine(str(args.data / "data_lmdb"), write=False)
    sums: dict[str, list] = {}
    contexts = {}
    forced = args.force_context or [None] * len(args.checkpoint)
    for label, ckpt, force in zip(labels, args.checkpoint, forced):
        model, ctx = load(ckpt, args.device)
        if force is not None:
            model.audio_context_patches = force
            ctx = force
        contexts[label] = ctx
        n = 0
        for key, length in meta:
            if length < 200:
                continue
            rec = engine[key]
            frames = (length // 100) * 100
            gt = torch.from_numpy(rec["motioncode"]).float()[:frames]
            audio = torch.from_numpy(rec["audio"]).float()
            audio = torch.nn.functional.pad(audio, (0, max(0, frames * 640 - audio.shape[0])))[: frames * 640]
            gt_verts = verts_of(gt)
            gt_vel = (gt_verts[1:] - gt_verts[:-1]).norm(dim=-1).mean()
            torch.manual_seed(0)
            gen = model.inference(audio[None].to(args.device))["pred_motion_code"][0].float().cpu()
            if gen.shape[0] != frames:
                raise RuntimeError(f"{label}: {gen.shape[0]} frames for {frames} expected")
            gv = verts_of(gen)
            lve, mhd, fdd = calc_val_metrics(gv[None], gt_verts[None])
            vel = (gv[1:] - gv[:-1]).norm(dim=-1).mean()
            sums.setdefault(label, []).append({
                "lve_mm": lve.item(), "mhd_mm": mhd.item(), "fdd": fdd.item(),
                "vel_ratio": (vel / gt_vel).item(),
                "head_sway": float(gen[:, 100:103].std(dim=0).mean()),
            })
            n += 1
            if args.max_clips and n >= args.max_clips:
                break
        del model
        torch.cuda.empty_cache()
        print(f"{label}: {n} clips, audio context {contexts[label]} chunks", flush=True)
    engine.close()

    header = ["model", "lve_mm", "mhd_mm", "fdd", "vel_ratio", "head_sway"]
    print(" ".join(f"{h:>10}" for h in header))
    for label, rows in sums.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{label:>10} " + " ".join(f"{m[k]:10.4f}" for k in header[1:]))


if __name__ == "__main__":
    main()
