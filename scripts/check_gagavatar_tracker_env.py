#!/usr/bin/env python

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TRACKER_CHECK = r"""
import sys

import torch
import torch._dynamo
import torchvision
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine
from core.libs.GAGAvatar_track.engines.human_matting import StyleMatteEngine

torch._dynamo.config.suppress_errors = True
print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"torchvision={torchvision.__version__}")
TrackEngine(focal_length=12.0, device="cpu")
StyleMatteEngine(device="cpu")._init_models()
"""

CUDA_RASTERIZER_CHECK = r"""
import torch
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

verts = torch.tensor(
    [[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]]],
    device="cuda",
    dtype=torch.float32,
)
faces = torch.tensor([[[0, 1, 2]]], device="cuda", dtype=torch.int64)
rasterize_meshes(Meshes(verts=verts, faces=faces), image_size=4)
print("pytorch3d_cuda_rasterizer=ok")
"""


def main():
    parser = argparse.ArgumentParser(description="Check the GAGAvatar tracker environment.")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GAGAVATAR_REPO", str(REPO_ROOT / "GAGAvatar")),
        help="GAGAvatar checkout path; defaults to ./GAGAvatar",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("GAGAVATAR_PYTHON", sys.executable),
        help="Python executable for the dedicated tracker environment",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also verify PyTorch3D CUDA rasterization",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve()
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(repo_path.exists(), f"GAGAvatar repo not found: {repo_path}")
    for asset in [
        "core/libs/GAGAvatar_track/assets/flame/FLAME_with_eye.pt",
        "core/libs/GAGAvatar_track/assets/emica/EMICA-CVT_flame2020_notexture.pt",
        "core/libs/GAGAvatar_track/assets/matting/stylematte_synth.pt",
    ]:
        require((repo_path / asset).exists(), f"missing tracker asset: {asset}")

    if failures:
        print("GAGAvatar tracker environment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["TORCHDYNAMO_DISABLE"] = "1"
    result = subprocess.run(
        [args.python, "-c", TRACKER_CHECK],
        cwd=repo_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return result.returncode

    if args.require_cuda:
        result = subprocess.run(
            [args.python, "-c", CUDA_RASTERIZER_CHECK],
            cwd=repo_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            return result.returncode

    print("GAGAvatar tracker environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
