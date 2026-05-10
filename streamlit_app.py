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
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_source_track,
    create_video_source_track,
    webrtc_streamer,
)

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
from app.realtime_pipeline import ARTalkPipeline


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

# Per-session pipeline; persisted across Streamlit reruns via
# session_state so the streamer / smoother / queues / worker thread
# stay alive between widget interactions.
if "pipeline" not in st.session_state:
    st.session_state.pipeline = ARTalkPipeline(
        model=model,
        flame_model=flame_model,
        mesh_renderer=mesh_renderer,
        device=args.device,
    )
pipeline = st.session_state.pipeline


def audio_frame_callback(frame):
    pipeline.push_audio_frame(frame)
    return frame


def on_audio_ended():
    """Inbound audio track ended (browser stopped sending) → user
    has ended the session. Tear down the pipeline (worker thread,
    queues, model state) and drop it from session_state so the next
    Start creates a fresh one.
    """
    pipeline.stop()
    if "pipeline" in st.session_state:
        del st.session_state["pipeline"]


video_source_track = create_video_source_track(
    pipeline.video_source_callback,
    key="artalk_video_source",
    fps=25,
)
audio_source_track = create_audio_source_track(
    pipeline.audio_source_callback,
    key="artalk_audio_source",
    sample_rate=48000,
    ptime=0.020,
)


def on_change():
    ctx = st.session_state["artalk-render"]
    if not ctx.state.playing and not ctx.state.signalling:
        video_source_track.stop()
        audio_source_track.stop()


# Inbound audio (browser mic → server). SENDONLY so the browser does
# not receive audio back on this peer connection — the synced delayed
# audio is delivered on the render-side connection below.
audio_ctx = webrtc_streamer(
    key="artalk-audio",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    on_audio_ended=on_audio_ended,
    media_stream_constraints={"audio": True, "video": False},
    on_change=on_change,
)

# Outbound audio + video (server → browser). The audio source emits
# the same input samples that produced the currently-rendered video
# frames, so they share the model's ~4 s latency and stay in sync.
webrtc_streamer(
    key="artalk-render",
    mode=WebRtcMode.RECVONLY,
    source_video_track=video_source_track,
    source_audio_track=audio_source_track,
    media_stream_constraints={"audio": True, "video": True},
    desired_playing_state=audio_ctx.state.playing,
)
