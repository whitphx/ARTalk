#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Streamlit + streamlit-webrtc realtime demo for ARTalk.

Browser microphone → ARTalk streaming inference → mesh-rendered avatar
video → browser. Phase 4 of the realtime work; see
``docs/realtime.md``.

Run on a GPU host that has ``./assets/ARTalk_wav2vec.pt``,
``./assets/config.json``, and ``./assets/FLAME_with_eye.pt``::

    streamlit run streamlit_app.py [-- --device cuda]

Browser microphone access requires a secure context (HTTPS or
``http://localhost``). For a remote GPU, the simplest setup is
SSH port forwarding::

    ssh -L 8501:localhost:8501 <gpu-host>

then open ``http://localhost:8501`` on the workstation.
"""

import argparse
import json

import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
from app.realtime_pipeline import ARTalkPipeline, ARTalkVideoTrack


@st.cache_resource
def load_model_and_renderer(device):
    audio_encoder = "wav2vec"
    ckpt = torch.load(
        f"./assets/ARTalk_{audio_encoder}.pt",
        map_location="cpu",
        weights_only=True,
    )
    configs = json.load(open("./assets/config.json"))
    configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
    model = BitwiseARModel(configs).to(device)
    model.train(False)
    model.load_state_dict(ckpt, strict=True)
    flame = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device)
    mesh = RenderMesh(image_size=512, faces=flame.get_faces(), scale=1.0)
    return model, flame, mesh


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", type=str)
# streamlit forwards extra CLI args after `--` separator.
args, _ = parser.parse_known_args()


st.set_page_config(page_title="ARTalk Realtime", page_icon=":speech_balloon:")
st.title("ARTalk Realtime")
st.caption(
    "Speak into the microphone — the avatar starts moving "
    "~4 seconds later (model chunk floor)."
)

model, flame_model, mesh_renderer = load_model_and_renderer(args.device)

# Per-session pipeline + paired video track. Persisted across
# Streamlit reruns via session_state so the streamer / smoother /
# queues stay alive between widget interactions.
if "pipeline" not in st.session_state:
    pipeline = ARTalkPipeline(
        model=model,
        flame_model=flame_model,
        mesh_renderer=mesh_renderer,
        device=args.device,
    )
    st.session_state.pipeline = pipeline
    st.session_state.video_track = ARTalkVideoTrack(pipeline)
pipeline = st.session_state.pipeline
video_track = st.session_state.video_track


def audio_frame_callback(frame):
    pipeline.push_audio_frame(frame)
    return frame


webrtc_streamer(
    key="artalk",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_frame_callback,
    source_video_track=video_track,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)
