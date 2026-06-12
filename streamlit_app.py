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
import asyncio
import base64
import json
import logging
import os
import threading
import time
from typing import Optional

import av
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_audio_source_track,
    create_video_source_track,
    webrtc_streamer,
)
from streamlit_webrtc.shutdown import SessionShutdownObserver

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
from app.realtime_pipeline import ARTalkPipeline

logger = logging.getLogger(__name__)

OPENAI_REALTIME_SAMPLE_RATE = 24000
DEFAULT_APPEARANCE = "mesh"
DEFAULT_STYLE = "default"
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_VOICE = "alloy"
REALTIME_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
]
DEFAULT_REALTIME_INSTRUCTIONS = (
    "You are speaking through an ARTalk avatar. Keep responses concise "
    "and conversational."
)


@st.cache_resource
def load_model_and_renderer(device, render_res):
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
    mesh = RenderMesh(image_size=render_res, faces=flame.get_faces(), scale=1.0)
    return model, flame, mesh


@st.cache_data
def list_gagavatar_ids():
    try:
        tracked = torch.load(
            "./assets/GAGAvatar/tracked.pt",
            map_location="cpu",
            weights_only=False,
        )
    except FileNotFoundError:
        return []
    return sorted(tracked.keys())


@st.cache_data
def list_style_ids():
    style_dir = "./assets/style_motion"
    if not os.path.isdir(style_dir):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(style_dir)
        if name.endswith(".pt")
    )


@st.cache_data
def load_style_motion(style_id):
    if style_id == DEFAULT_STYLE:
        return None
    style_motion = torch.load(
        f"./assets/style_motion/{style_id}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(style_motion.shape) != (50, 106):
        raise ValueError(f"Invalid style motion shape: {tuple(style_motion.shape)}")
    return style_motion


@st.cache_resource
def load_gagavatar(device):
    from app.GAGAvatar import GAGAvatar

    gagavatar = GAGAvatar().to(device)
    gagavatar_flame = FLAMEModel(
        n_shape=300,
        n_exp=100,
        scale=5.0,
        no_lmks=True,
    ).to(device)
    return gagavatar, gagavatar_flame


class OpenAIRealtimeBridge:
    """Bridge browser mic audio to OpenAI and feed response audio to ARTalk."""

    def __init__(
        self,
        *,
        api_key: str,
        pipeline: ARTalkPipeline,
        model: str,
        voice: str,
        instructions: str,
    ) -> None:
        self._api_key = api_key
        self._pipeline = pipeline
        self._model = model
        self._voice = voice
        self._instructions = instructions

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: Optional["asyncio.Queue[bytes]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=OPENAI_REALTIME_SAMPLE_RATE
        )

        self._state_lock = threading.Lock()
        self._connected = False
        self._error: Optional[str] = None
        self._user_transcript = ""
        self._assistant_transcript = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._ready_event.clear()
        with self._state_lock:
            self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="OpenAIRealtimeBridge",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=3.0)

    def wait_until_connected(self, timeout: float) -> bool:
        # Use a small sleep loop instead of exposing the asyncio event across
        # threads; Streamlit calls this only while pre-warming Interactive mode.
        stop_at = time.monotonic() + timeout
        while time.monotonic() < stop_at:
            with self._state_lock:
                if self._connected:
                    return True
                if self._error:
                    return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_event.set)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._thread = None
        with self._state_lock:
            self._connected = False

    def push_input(self, frame: av.AudioFrame) -> None:
        loop, q = self._loop, self._input_queue
        if loop is None or q is None or loop.is_closed():
            return
        for resampled in self._resampler.resample(frame):
            arr = resampled.to_ndarray()
            pcm = arr.astype(np.int16, copy=False).tobytes()
            if not pcm:
                continue
            try:
                loop.call_soon_threadsafe(self._queue_input, pcm)
            except RuntimeError:
                return

    def _queue_input(self, pcm: bytes) -> None:
        if self._input_queue is None:
            return
        try:
            self._input_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "user": self._user_transcript,
                "assistant": self._assistant_transcript,
            }

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._input_queue = asyncio.Queue(maxsize=256)
            self._stop_event = asyncio.Event()
            self._ready_event.set()
            loop.run_until_complete(self._session())
        except Exception as exc:
            logger.exception("OpenAI Realtime bridge crashed")
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._state_lock:
                self._connected = False
            try:
                loop.close()
            finally:
                self._loop = None

    async def _session(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI SDK to use Interactive mode: pip install openai"
            ) from exc

        if self._stop_event is None or self._input_queue is None:
            raise RuntimeError("Realtime bridge loop is not initialized")

        client = AsyncOpenAI(api_key=self._api_key)
        async with client.realtime.connect(model=self._model) as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "model": self._model,
                    "instructions": self._instructions,
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": self._voice},
                    },
                }
            )
            with self._state_lock:
                self._connected = True

            tasks = [
                asyncio.create_task(self._send_loop(conn), name="openai-send"),
                asyncio.create_task(self._recv_loop(conn), name="openai-recv"),
                asyncio.create_task(self._stop_event.wait(), name="openai-stop"),
            ]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_loop(self, conn) -> None:
        if self._input_queue is None:
            return
        while True:
            pcm = await self._input_queue.get()
            await conn.input_audio_buffer.append(
                audio=base64.b64encode(pcm).decode("ascii")
            )

    async def _recv_loop(self, conn) -> None:
        async for event in conn:
            etype = getattr(event, "type", "")
            if etype == "response.output_audio.delta":
                self._push_response_audio(base64.b64decode(event.delta))
            elif etype == "response.output_audio_transcript.delta":
                with self._state_lock:
                    self._assistant_transcript += getattr(event, "delta", "") or ""
            elif etype == "response.done":
                with self._state_lock:
                    if (
                        self._assistant_transcript
                        and not self._assistant_transcript.endswith("\n")
                    ):
                        self._assistant_transcript += "\n"
            elif etype == "conversation.item.input_audio_transcription.delta":
                with self._state_lock:
                    self._user_transcript += getattr(event, "delta", "") or ""
            elif etype == "conversation.item.input_audio_transcription.completed":
                with self._state_lock:
                    if self._user_transcript and not self._user_transcript.endswith(
                        "\n"
                    ):
                        self._user_transcript += "\n"
            elif etype == "error":
                err = getattr(event, "error", None)
                msg = getattr(err, "message", None) or repr(err)
                logger.warning("OpenAI Realtime API error: %s", msg)
                with self._state_lock:
                    self._error = msg

    def _push_response_audio(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = OPENAI_REALTIME_SAMPLE_RATE
        self._pipeline.push_audio_frame(frame)


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--render-res", default=512, type=int)
# streamlit forwards extra CLI args after `--` separator.
args, _ = parser.parse_known_args()


st.set_page_config(page_title="ARTalk Realtime", page_icon=":speech_balloon:")
st.title("ARTalk Realtime")
st.caption(
    "Speak into the microphone — the avatar starts moving "
    "~4 seconds later (model chunk floor)."
)

model, flame_model, mesh_renderer = load_model_and_renderer(
    args.device,
    args.render_res,
)

with st.sidebar:
    gagavatar_ids = list_gagavatar_ids()
    style_ids = list_style_ids()
    appearance = st.selectbox(
        "Appearance",
        [DEFAULT_APPEARANCE] + gagavatar_ids,
        index=0,
    )
    default_style_index = (
        style_ids.index("natural_0") + 1 if "natural_0" in style_ids else 0
    )
    style_id = st.selectbox(
        "Style",
        [DEFAULT_STYLE] + style_ids,
        index=default_style_index,
    )
    mode = st.radio("Mode", ["Loopback", "Interactive"], horizontal=True)
    api_key = ""
    realtime_model = DEFAULT_REALTIME_MODEL
    realtime_voice = DEFAULT_REALTIME_VOICE
    realtime_instructions = DEFAULT_REALTIME_INSTRUCTIONS
    if mode == "Interactive":
        st.header("OpenAI Realtime")
        api_key = st.text_input(
            "OPENAI_API_KEY",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Used only by the Streamlit server.",
        )
        realtime_model = st.text_input("Model", value=DEFAULT_REALTIME_MODEL)
        realtime_voice = st.selectbox(
            "Voice",
            REALTIME_VOICES,
            index=REALTIME_VOICES.index(DEFAULT_REALTIME_VOICE),
        )
        realtime_instructions = st.text_area(
            "Instructions",
            value=DEFAULT_REALTIME_INSTRUCTIONS,
            height=120,
        )


PIPELINE_KEY = "artalk_pipeline"
PIPELINE_CONFIG_KEY = "artalk_pipeline_config"
BRIDGE_KEY = "openai_realtime_bridge"
BRIDGE_CONFIG_KEY = "openai_realtime_bridge_config"
BRIDGE_SHUTDOWN_OBSERVER_KEY = "openai_realtime_bridge_shutdown_observer"


def stop_bridge() -> None:
    bridge = st.session_state.pop(BRIDGE_KEY, None)
    if bridge is not None:
        bridge.stop()
    observer = st.session_state.pop(BRIDGE_SHUTDOWN_OBSERVER_KEY, None)
    if isinstance(observer, SessionShutdownObserver):
        observer.stop()
    st.session_state.pop(BRIDGE_CONFIG_KEY, None)


def get_pipeline() -> ARTalkPipeline:
    renderer_mode = "mesh" if appearance == DEFAULT_APPEARANCE else "gagavatar"
    style_motion = load_style_motion(style_id)
    render_res = args.render_res if renderer_mode == "mesh" else 512
    gagavatar = None
    gagavatar_flame = None
    if renderer_mode == "gagavatar":
        gagavatar, gagavatar_flame = load_gagavatar(args.device)
    config = (
        args.device,
        mode,
        render_res,
        appearance,
        style_id,
    )
    pipeline = st.session_state.get(PIPELINE_KEY)
    if pipeline is not None and st.session_state.get(PIPELINE_CONFIG_KEY) != config:
        pipeline.stop()
        pipeline = None
    if pipeline is None:
        pipeline = ARTalkPipeline(
            model=model,
            flame_model=flame_model,
            mesh_renderer=mesh_renderer,
            device=args.device,
            style_motion=style_motion,
            render_res=render_res,
            renderer_mode=renderer_mode,
            gagavatar=gagavatar,
            gagavatar_flame=gagavatar_flame,
            shape_id=appearance if renderer_mode == "gagavatar" else None,
        )
        st.session_state[PIPELINE_KEY] = pipeline
        st.session_state[PIPELINE_CONFIG_KEY] = config
    return pipeline


def stop_pipeline() -> None:
    pipeline = st.session_state.pop(PIPELINE_KEY, None)
    if pipeline is not None:
        pipeline.stop()
    st.session_state.pop(PIPELINE_CONFIG_KEY, None)


if mode == "Loopback":
    stop_bridge()

if mode == "Interactive" and not api_key:
    stop_bridge()
    if st.session_state.get(PIPELINE_CONFIG_KEY) != (
        args.device,
        mode,
        args.render_res,
    ):
        stop_pipeline()
    st.info("Enter `OPENAI_API_KEY` in the sidebar to use Interactive mode.")
    st.stop()


try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"Failed to initialize ARTalk avatar pipeline: {exc}")
    st.stop()


def get_bridge() -> OpenAIRealtimeBridge:
    config = (
        api_key,
        realtime_model,
        realtime_voice,
        realtime_instructions,
        id(pipeline),
    )
    bridge = st.session_state.get(BRIDGE_KEY)
    if bridge is not None and st.session_state.get(BRIDGE_CONFIG_KEY) != config:
        stop_bridge()
        bridge = None
    if bridge is None:
        bridge = OpenAIRealtimeBridge(
            api_key=api_key,
            pipeline=pipeline,
            model=realtime_model,
            voice=realtime_voice,
            instructions=realtime_instructions,
        )
        st.session_state[BRIDGE_KEY] = bridge
        st.session_state[BRIDGE_CONFIG_KEY] = config
        st.session_state[BRIDGE_SHUTDOWN_OBSERVER_KEY] = SessionShutdownObserver(
            bridge.stop
        )
    return bridge


bridge = get_bridge() if mode == "Interactive" else None
if bridge is not None and not bridge.is_running:
    with st.spinner("Connecting to OpenAI Realtime..."):
        bridge.start()
        bridge.wait_until_connected(timeout=8.0)
    snap = bridge.snapshot()
    if snap["error"]:
        st.error(f"OpenAI Realtime API error: {snap['error']}")
    elif not snap["connected"]:
        st.warning("OpenAI Realtime is still connecting. Wait a moment before START.")


def on_loopback_audio_frame(frame: av.AudioFrame) -> None:
    pipeline.push_audio_frame(frame)


def on_interactive_audio_frame(frame: av.AudioFrame) -> None:
    if bridge is not None:
        bridge.push_input(frame)


def on_audio_ended() -> None:
    stop_bridge()
    stop_pipeline()


video_source_track = create_video_source_track(
    pipeline.video_source_callback,
    key=f"artalk_video_source_{mode.lower()}",
    fps=25,
)
audio_source_track = create_audio_source_track(
    pipeline.audio_source_callback,
    key=f"artalk_audio_source_{mode.lower()}",
    sample_rate=16000,
    ptime=0.020,
)
audio_sink_track = create_audio_sink_track(
    on_loopback_audio_frame if mode == "Loopback" else on_interactive_audio_frame,
    key=f"artalk_audio_sink_{mode.lower()}",
    on_ended=on_audio_ended,
)

streamer_key = f"artalk_{mode.lower()}"


def on_change() -> None:
    ctx = st.session_state.get(streamer_key)
    if ctx is None:
        return
    if mode == "Interactive" and ctx.state.playing and bridge is not None:
        bridge.start()
    if not ctx.state.playing and not ctx.state.signalling:
        if bridge is not None:
            bridge.stop()
        video_source_track.stop()
        audio_source_track.stop()


webrtc_streamer(
    key=streamer_key,
    mode=WebRtcMode.SENDRECV,
    source_video_track=video_source_track,
    source_audio_track=audio_source_track,
    sink_audio_track=audio_sink_track,
    media_stream_constraints={"audio": True, "video": False},
    on_change=on_change,
)


if bridge is not None:

    @st.fragment(run_every="500ms")
    def render_interactive_status() -> None:
        snap = bridge.snapshot()
        if snap["error"]:
            st.error(f"OpenAI Realtime API error: {snap['error']}")
        if snap["user"] or snap["assistant"]:
            st.subheader("Transcript")
            user_col, assistant_col = st.columns(2)
            with user_col:
                st.caption("You")
                st.text(snap["user"] or "-")
            with assistant_col:
                st.caption("Assistant")
                st.text(snap["assistant"] or "-")

    render_interactive_status()
