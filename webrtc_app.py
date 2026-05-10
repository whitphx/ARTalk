#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""WebRTC realtime demo for ARTalk.

Browser microphone → ARTalk streaming inference → mesh-rendered
avatar video → browser, over WebRTC. Phase 4 of the realtime work;
see ``docs/realtime.md`` Phase 4 for the design rationale.

Run on a GPU host that has ``./assets/ARTalk_wav2vec.pt``,
``./assets/config.json``, and ``./assets/FLAME_with_eye.pt``::

    python webrtc_app.py [--device cuda] [--port 8000] [--host 0.0.0.0]

FastRTC's built-in dev UI is exposed by ``stream.ui.launch``; open
it in a browser, click the WebRTC start button, and speak — the
avatar video starts ~4 seconds later (the model's chunk floor).

Browser microphone access requires a secure context (HTTPS or
``http://localhost``). Two practical setups for this:

* **SSH port forwarding (recommended for dev)**: leave the server
  on plain HTTP and forward the port from your workstation —
  ``ssh -L 8000:localhost:8000 <gpu-host>`` — then open
  ``http://localhost:8000``. The browser treats localhost as
  secure.
* **HTTPS** with ``--ssl-keyfile`` / ``--ssl-certfile`` for
  reachable / shared deployments. Self-signed certs work for
  testing but produce browser warnings.
"""

import argparse
import json

import torch
from fastrtc import Stream

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
from app.webrtc import ARTalkHandler


def load_model(device):
    audio_encoder = "wav2vec"
    ckpt = torch.load(
        f"./assets/ARTalk_{audio_encoder}.pt", map_location="cpu", weights_only=True
    )
    configs = json.load(open("./assets/config.json"))
    configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
    model = BitwiseARModel(configs).to(device)
    model.train(False)
    model.load_state_dict(ckpt, strict=True)
    return model


def load_flame(device):
    flame = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device)
    mesh = RenderMesh(image_size=512, faces=flame.get_faces(), scale=1.0)
    return flame, mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--concurrency-limit",
        default=1,
        type=int,
        help="max concurrent browser connections — keep at 1 for a single GPU.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        type=str,
        help="path to TLS private key. Required (with --ssl-certfile) for "
             "HTTPS serving so browser microphone access works on a remote host.",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        type=str,
        help="path to TLS certificate.",
    )
    parser.add_argument(
        "--ssl-keyfile-password",
        default=None,
        type=str,
        help="optional password for an encrypted --ssl-keyfile.",
    )
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.device)
    print("Loading FLAME / renderer...")
    flame_model, mesh_renderer = load_flame(args.device)

    handler_template = ARTalkHandler(
        model=model,
        flame_model=flame_model,
        mesh_renderer=mesh_renderer,
        device=args.device,
    )

    stream = Stream(
        handler=handler_template,
        mode="send-receive",
        modality="audio-video",
        concurrency_limit=args.concurrency_limit,
    )
    stream.ui.launch(
        server_name=args.host,
        server_port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile_password=args.ssl_keyfile_password,
    )


if __name__ == "__main__":
    main()
