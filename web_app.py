#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import subprocess
import threading
import traceback
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS

from app.avatar_registry import UPLOADED_AVATAR_ROOT, available_avatars
from app.gagavatar_video import (
    GAGAVATAR_RENDER_MODE,
    MESH_RENDER_MODE,
    check_gagavatar_render_environment,
    get_gagavatar_video_renderer,
)
from app.gagavatar_tracking import check_tracker_environment, track_uploaded_avatar
from app.web_inference import (
    available_styles,
    get_web_inference_service,
    write_web_metadata,
    write_web_result,
)


JOB_ROOT = Path("render_results/web_jobs").resolve()
AVATAR_JOB_ROOT = UPLOADED_AVATAR_ROOT
FRONTEND_DIST = Path("frontend/dist")
GTTS_LANG = {
    "English": "en",
    "中文": "zh",
    "日本語": "ja",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es",
}

app = FastAPI(title="ARTalk Web Renderer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def job_dir(job_id):
    path = JOB_ROOT / job_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return path


def avatar_job_dir(avatar_id):
    if not avatar_id or any(char not in "0123456789abcdef" for char in avatar_id):
        raise HTTPException(status_code=404, detail="Avatar job not found")
    path = AVATAR_JOB_ROOT / avatar_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Avatar job not found")
    return path


def write_state(path, state):
    path.mkdir(parents=True, exist_ok=True)
    tmp_path = path / f".state.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    tmp_path.replace(path / "state.json")


def read_state(path):
    with open(path / "state.json") as f:
        return json.load(f)


def start_thread(target, **kwargs):
    thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
    thread.start()
    return thread


def write_text_audio(text, language, output_dir):
    mp3_path = output_dir / "input.mp3"
    wav_path = output_dir / "input.wav"
    gTTS(text=text, lang=GTTS_LANG[language]).save(str(mp3_path))
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="Text input requires ffmpeg to convert generated speech to WAV.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to convert generated speech to WAV: {exc.stderr.strip()}",
        ) from exc
    return wav_path


def run_job(job_id, *, input_path, style_id, clip_length, device, avatar_id, render_mode):
    path = JOB_ROOT / job_id
    try:
        write_state(path, {"id": job_id, "status": "running", "stage": "loading model"})
        service = get_web_inference_service(device)
        write_state(path, {"id": job_id, "status": "running", "stage": "generating motion"})
        result = service.generate(
            input_path,
            style_id=style_id,
            clip_length=clip_length,
            avatar_id=avatar_id,
        )
        write_state(path, {"id": job_id, "status": "running", "stage": "writing mesh"})
        metadata = write_web_result(result, path)
        if render_mode == GAGAVATAR_RENDER_MODE:
            write_state(path, {"id": job_id, "status": "running", "stage": "rendering colored video"})
            renderer = get_gagavatar_video_renderer(device)
            video_path = renderer.render_video(result, path, avatar_id=avatar_id)
            metadata["renderMode"] = GAGAVATAR_RENDER_MODE
            metadata["videoUrl"] = video_path.name
            write_web_metadata(metadata, path)
        write_state(
            path,
            {
                "id": job_id,
                "status": "complete",
                "stage": "complete",
                "metadata": f"/api/jobs/{job_id}/metadata",
                "frameCount": metadata["frameCount"],
            },
        )
    except Exception as exc:
        write_state(
            path,
            {
                "id": job_id,
                "status": "failed",
                "stage": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def run_avatar_job(avatar_id, *, input_path, device):
    path = AVATAR_JOB_ROOT / avatar_id
    try:
        write_state(path, {"id": avatar_id, "status": "running", "stage": "tracking face"})
        track_uploaded_avatar(input_path, path, device=device)
        write_state(
            path,
            {
                "id": avatar_id,
                "avatarId": f"uploaded:{avatar_id}",
                "status": "complete",
                "stage": "complete",
            },
        )
    except Exception as exc:
        write_state(
            path,
            {
                "id": avatar_id,
                "status": "failed",
                "stage": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


@app.get("/api/config")
def config():
    return {
        "styles": ["default"] + available_styles(),
        "avatars": available_avatars(),
        "languages": list(GTTS_LANG.keys()),
        "defaultStyle": "natural_0" if "natural_0" in available_styles() else "default",
        "defaultAvatar": "mesh",
        "renderModes": [
            {"id": MESH_RENDER_MODE, "label": "Browser mesh"},
            {"id": GAGAVATAR_RENDER_MODE, "label": "Colored video (server)"},
        ],
        "defaultRenderMode": MESH_RENDER_MODE,
    }


@app.get("/api/avatars")
def list_avatars():
    return {"avatars": available_avatars()}


@app.post("/api/avatar-jobs")
async def create_avatar_job(
    device: str = Form("auto"),
    image_file: UploadFile = File(...),
):
    suffix = Path(image_file.filename or "avatar.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Avatar image must be a JPG or PNG")
    try:
        check_tracker_environment()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    avatar_id = uuid.uuid4().hex
    path = AVATAR_JOB_ROOT / avatar_id
    path.mkdir(parents=True, exist_ok=True)
    input_path = path / f"upload{suffix}"
    with open(input_path, "wb") as f:
        f.write(await image_file.read())
    state = {"id": avatar_id, "status": "queued", "stage": "queued"}
    write_state(path, state)
    start_thread(
        run_avatar_job,
        avatar_id=avatar_id,
        input_path=str(input_path),
        device=device,
    )
    return state


@app.get("/api/avatar-jobs/{avatar_id}")
def get_avatar_job(avatar_id: str):
    return read_state(avatar_job_dir(avatar_id))


@app.get("/api/avatar-jobs/{avatar_id}/preview.jpg")
def get_avatar_preview(avatar_id: str):
    file_path = avatar_job_dir(avatar_id) / "preview.jpg"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Avatar preview not ready")
    return FileResponse(file_path)


@app.post("/api/jobs")
async def create_job(
    input_type: Literal["audio", "text"] = Form("audio"),
    style_id: str = Form("default"),
    clip_length: int = Form(750),
    device: str = Form("auto"),
    avatar_id: str = Form("mesh"),
    render_mode: Literal["mesh", "gagavatar"] = Form(MESH_RENDER_MODE),
    text: str | None = Form(None),
    text_language: str = Form("English"),
    audio_file: UploadFile | None = File(None),
):
    if render_mode == GAGAVATAR_RENDER_MODE:
        if avatar_id == "mesh":
            raise HTTPException(
                status_code=400,
                detail="Choose a registered or built-in GAGAvatar avatar for colored video output",
            )
        try:
            check_gagavatar_render_environment(device)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex
    path = JOB_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    if input_type == "text":
        if text is None or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        if text_language not in GTTS_LANG:
            raise HTTPException(status_code=400, detail="Unsupported text language")
        input_path = write_text_audio(text, text_language, path)
    else:
        if audio_file is None:
            raise HTTPException(status_code=400, detail="Audio file is required")
        suffix = Path(audio_file.filename or "input.wav").suffix or ".wav"
        input_path = path / f"input{suffix}"
        with open(input_path, "wb") as f:
            f.write(await audio_file.read())
    state = {"id": job_id, "status": "queued", "stage": "queued"}
    write_state(path, state)
    start_thread(
        run_job,
        job_id=job_id,
        input_path=str(input_path),
        style_id=style_id,
        clip_length=clip_length,
        device=device,
        avatar_id=avatar_id,
        render_mode=render_mode,
    )
    return state


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return read_state(job_dir(job_id))


@app.get("/api/jobs/{job_id}/metadata")
def get_metadata(job_id: str):
    path = job_dir(job_id)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not ready")
    with open(metadata_path) as f:
        metadata = json.load(f)
    return {
        **metadata,
        "verticesUrl": f"/api/jobs/{job_id}/vertices.f32",
        "facesUrl": f"/api/jobs/{job_id}/faces.i32",
        "regionLabelsUrl": (
            f"/api/jobs/{job_id}/{metadata['regionLabelsUrl']}"
            if metadata.get("regionLabelsUrl")
            else None
        ),
        "audioUrl": f"/api/jobs/{job_id}/audio.wav",
        "motionsUrl": f"/api/jobs/{job_id}/motions.pt",
        "videoUrl": f"/api/jobs/{job_id}/{metadata['videoUrl']}" if metadata.get("videoUrl") else None,
    }


@app.get("/api/jobs/{job_id}/{name}")
def get_job_file(job_id: str, name: str):
    if name not in {"vertices.f32", "faces.i32", "regions.u8", "audio.wav", "motions.pt", "gagavatar.mp4"}:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = job_dir(job_id) / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(file_path)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
