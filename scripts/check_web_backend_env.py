#!/usr/bin/env python

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Check the ARTalk web backend environment.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also check CUDA colored-video dependencies",
    )
    args = parser.parse_args()

    failures = []
    info = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    info.append(f"python={sys.executable}")
    for module in ["fastapi", "numpy", "scipy", "torch", "torchaudio", "transformers", "uvicorn"]:
        require(importlib.util.find_spec(module) is not None, f"missing Python module: {module}")

    require(shutil.which("ffmpeg") is not None, "ffmpeg is not on PATH")

    try:
        import torch

        info.append(f"torch={torch.__version__}")
        info.append(f"torch_cuda={torch.version.cuda}")
        info.append(f"cuda_available={torch.cuda.is_available()}")
        if args.full:
            require(torch.cuda.is_available(), "CUDA is not available to the backend process")
    except Exception as exc:
        failures.append(f"failed to import/check torch: {exc}")

    for asset in [
        "assets/ARTalk_wav2vec.pt",
        "assets/config.json",
        "assets/FLAME_with_eye.pt",
        "assets/GAGAvatar/GAGAvatar.pt",
        "assets/GAGAvatar/tracked.pt",
    ]:
        require((REPO_ROOT / asset).exists(), f"missing asset: {asset}")

    gagavatar_repo = REPO_ROOT / "GAGAvatar"
    require(
        gagavatar_repo.exists(),
        "GAGAvatar submodule is missing; run `git submodule update --init --recursive`",
    )

    if args.full:
        require(
            importlib.util.find_spec("diff_gaussian_rasterization_32d") is not None,
            "missing diff_gaussian_rasterization_32d; run scripts/install_gagavatar_rasterizer.sh",
        )

    for line in info:
        print(line)
    if failures:
        print("\nEnvironment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("ARTalk web backend environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
