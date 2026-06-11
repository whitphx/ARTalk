#!/usr/bin/env python
"""Export GAGAvatar's 32-channel StyleUNet upsampler to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class UpsamplerExportWrapper(torch.nn.Module):
    def __init__(self, upsampler: torch.nn.Module):
        super().__init__()
        self.upsampler = upsampler

    def forward(self, raster_features: torch.Tensor) -> torch.Tensor:
        return self.upsampler(raster_features, randomize_noise=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frontend/public/models/gagavatar_upsampler.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from app.GAGAvatar import GAGAvatar

    device = torch.device(args.device)
    gagavatar = GAGAvatar().to(device).eval()
    model = UpsamplerExportWrapper(gagavatar.upsampler).to(device).eval()
    sample = torch.zeros(1, 32, 512, 512, dtype=torch.float32, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(args.output),
        input_names=["raster_features"],
        output_names=["rgb"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
