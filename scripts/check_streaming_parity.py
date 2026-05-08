#!/usr/bin/env python
"""Parity check for ARTalkStreamer.

Loads the released ARTalk wav2vec checkpoint, runs both the original
one-shot ``BitwiseARModel.inference`` and the streaming
``ARTalkStreamer`` over the same audio fed in arbitrary-sized
sub-chunks, and asserts the outputs match within a tight tolerance.

Run on a GPU host where the model and ``./assets`` are available:

    python scripts/check_streaming_parity.py [-a demo/eng1.wav] [--device cuda]
"""

import argparse
import json

import torch
import torchaudio

from app import BitwiseARModel
from app.streaming import ARTalkStreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", "-a", default="./demo/eng1.wav", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--feed-chunk-samples",
        type=int,
        default=4000,
        help="size of chunks fed to ARTalkStreamer.feed (samples @16kHz). "
             "Choose != patch_audio_length so buffering is exercised.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--style", default=None, type=str,
                        help="optional style id under assets/style_motion (e.g. natural_0)")
    args = parser.parse_args()

    device = args.device
    audio_encoder = "wav2vec"

    ckpt = torch.load(
        f"./assets/ARTalk_{audio_encoder}.pt", map_location="cpu", weights_only=True
    )
    configs = json.load(open("./assets/config.json"))
    configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
    model = BitwiseARModel(configs).to(device)
    model.train(False)
    model.load_state_dict(ckpt, strict=True)

    style_motion = None
    if args.style is not None:
        style_motion = torch.load(
            f"./assets/style_motion/{args.style}.pt", map_location="cpu", weights_only=True
        )

    audio, sr = torchaudio.load(args.audio)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0).to(device)
    print(f"audio: {audio.shape[0]} samples ({audio.shape[0] / 16000:.2f}s)")

    batch = {"audio": audio[None]}
    if style_motion is not None:
        batch["style_motion"] = style_motion[None].to(device)
    one_shot = model.inference(batch)[0]

    streamer = ARTalkStreamer(model, style_motion=style_motion)
    pieces = []
    for i in range(0, audio.shape[0], args.feed_chunk_samples):
        out = streamer.feed(audio[i : i + args.feed_chunk_samples])
        if out.shape[0] > 0:
            pieces.append(out)
    tail = streamer.finish()
    if tail.shape[0] > 0:
        pieces.append(tail)
    streamed = torch.cat(pieces, dim=0) if pieces else torch.zeros(
        0, model.basic_vae.motion_dim, device=device
    )

    print(f"one_shot: {tuple(one_shot.shape)}")
    print(f"streamed: {tuple(streamed.shape)}")
    assert one_shot.shape == streamed.shape, "shape mismatch"

    diff = (one_shot - streamed).abs()
    print(f"max abs diff:  {diff.max().item():.3e}")
    print(f"mean abs diff: {diff.mean().item():.3e}")
    if not torch.allclose(one_shot, streamed, atol=args.atol):
        raise SystemExit(
            f"streaming output diverges from one-shot beyond atol={args.atol}"
        )
    print("OK: streaming output matches one-shot inference.")


if __name__ == "__main__":
    main()
