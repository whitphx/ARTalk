#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import traceback
import uuid
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS

from app.web_inference import available_styles, get_web_inference_service, write_web_result


JOB_ROOT = Path("render_results/web_jobs")
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


def write_state(path, state):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "state.json", "w") as f:
        json.dump(state, f)


def read_state(path):
    with open(path / "state.json") as f:
        return json.load(f)


def run_job(job_id, *, input_path, style_id, clip_length, device):
    path = JOB_ROOT / job_id
    try:
        write_state(path, {"id": job_id, "status": "running", "stage": "loading model"})
        service = get_web_inference_service(device)
        write_state(path, {"id": job_id, "status": "running", "stage": "generating motion"})
        result = service.generate(input_path, style_id=style_id, clip_length=clip_length)
        write_state(path, {"id": job_id, "status": "running", "stage": "writing mesh"})
        metadata = write_web_result(result, path)
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


@app.get("/api/config")
def config():
    return {
        "styles": ["default"] + available_styles(),
        "languages": list(GTTS_LANG.keys()),
        "defaultStyle": "natural_0" if "natural_0" in available_styles() else "default",
    }


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    input_type: Literal["audio", "text"] = Form("audio"),
    style_id: str = Form("default"),
    clip_length: int = Form(750),
    device: str = Form("auto"),
    text: str | None = Form(None),
    text_language: str = Form("English"),
    audio_file: UploadFile | None = File(None),
):
    job_id = uuid.uuid4().hex
    path = JOB_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    if input_type == "text":
        if text is None or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        if text_language not in GTTS_LANG:
            raise HTTPException(status_code=400, detail="Unsupported text language")
        input_path = path / "input.mp3"
        gTTS(text=text, lang=GTTS_LANG[text_language]).save(str(input_path))
    else:
        if audio_file is None:
            raise HTTPException(status_code=400, detail="Audio file is required")
        suffix = Path(audio_file.filename or "input.wav").suffix or ".wav"
        input_path = path / f"input{suffix}"
        with open(input_path, "wb") as f:
            f.write(await audio_file.read())
    write_state(path, {"id": job_id, "status": "queued", "stage": "queued"})
    background_tasks.add_task(
        run_job,
        job_id,
        input_path=str(input_path),
        style_id=style_id,
        clip_length=clip_length,
        device=device,
    )
    return read_state(path)


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
        "audioUrl": f"/api/jobs/{job_id}/audio.wav",
        "motionsUrl": f"/api/jobs/{job_id}/motions.pt",
    }


@app.get("/api/jobs/{job_id}/{name}")
def get_job_file(job_id: str, name: str):
    if name not in {"vertices.f32", "faces.i32", "audio.wav", "motions.pt"}:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = job_dir(job_id) / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(file_path)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
