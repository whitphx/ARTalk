#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import platform
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import savgol_filter

from app.avatar_registry import get_avatar_shape_code
from app.models import BitwiseARModel
from app.flame_model.FLAME import FLAMEModel


STYLE_DIR = Path("assets/style_motion")


def available_styles():
    if not STYLE_DIR.exists():
        return []
    return sorted(path.stem for path in STYLE_DIR.glob("*.pt"))


def select_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        # ARTalk hits non-divisible adaptive pooling shapes that PyTorch's MPS
        # backend does not implement yet, so keep the macOS default predictable.
        return torch.device("cpu")
    return torch.device("cpu")



def smooth_motion_savgol(motion_codes):
    motion_np = motion_codes.clone().detach().cpu().numpy()
    motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
    motion_np_smoothed[..., 100:103] = savgol_filter(
        motion_np[..., 100:103], window_length=9, polyorder=3, axis=0
    )
    return torch.tensor(motion_np_smoothed).type_as(motion_codes)


@dataclass
class WebInferenceResult:
    audio: torch.Tensor
    motions: torch.Tensor
    vertices: np.ndarray
    faces: np.ndarray
    sample_rate: int
    fps: int
    avatar_id: str


class WebInferenceService:
    def __init__(self, device="auto"):
        self.device = select_device(device)
        self._lock = threading.Lock()
        audio_encoder = "wav2vec"
        ckpt = torch.load(
            f"./assets/ARTalk_{audio_encoder}.pt",
            map_location="cpu",
            weights_only=True,
        )
        configs = json.load(open("./assets/config.json"))
        configs["AR_CONFIG"]["AUDIO_ENCODER"] = audio_encoder
        self.model = BitwiseARModel(configs).eval().to(self.device)
        self.model.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(
            n_shape=300,
            n_exp=100,
            scale=1.0,
            no_lmks=True,
        ).to(self.device)
        self.style_motion = None

    def set_style_motion(self, style_id):
        if style_id in (None, "", "default"):
            self.style_motion = None
            return
        style_motion = torch.load(
            STYLE_DIR / f"{style_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if tuple(style_motion.shape) != (50, 106):
            raise ValueError(f"Invalid style motion shape for {style_id}: {style_motion.shape}")
        self.style_motion = style_motion[None].to(self.device)

    @torch.no_grad()
    def generate(self, audio_path, *, style_id="default", clip_length=750, avatar_id="mesh"):
        with self._lock:
            audio, sr = torchaudio.load(audio_path)
            audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
            self.set_style_motion(style_id)
            audio_batch = {
                "audio": audio[None].to(self.device),
                "style_motion": self.style_motion,
            }
            pred_motions = self.model.inference(audio_batch, with_gtmotion=False)[0]
            pred_motions = smooth_motion_savgol(pred_motions)[:clip_length]
            pred_motions[..., 104:] *= 0.0
            shape_code = get_avatar_shape_code(avatar_id)
            if shape_code is None:
                shape_code = audio.new_zeros(1, 300)
            shape_code = shape_code.to(self.device).expand(
                pred_motions.shape[0],
                -1,
            )
            vertices = self.model.basic_vae.get_flame_verts(
                self.flame_model,
                shape_code,
                pred_motions,
                with_global=True,
            )
            audio = audio[: int(vertices.shape[0] / 25.0 * 16000)]
            return WebInferenceResult(
                audio=audio.float().cpu(),
                motions=pred_motions.float().cpu(),
                vertices=vertices.float().cpu().numpy().astype(np.float32, copy=False),
                faces=self.flame_model.get_faces().cpu().numpy().astype(np.int32, copy=False),
                sample_rate=16000,
                fps=25,
                avatar_id=avatar_id,
            )


_services = {}
_services_lock = threading.Lock()


def get_web_inference_service(device="auto"):
    resolved = str(select_device(device))
    with _services_lock:
        if resolved not in _services:
            _services[resolved] = WebInferenceService(resolved)
        return _services[resolved]


def write_web_result(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.vertices.tofile(output_dir / "vertices.f32")
    result.faces.tofile(output_dir / "faces.i32")
    torch.save(result.motions, output_dir / "motions.pt")
    torchaudio.save(
        str(output_dir / "audio.wav"),
        result.audio[None],
        result.sample_rate,
    )
    metadata = {
        "renderMode": "mesh",
        "fps": result.fps,
        "sampleRate": result.sample_rate,
        "frameCount": int(result.vertices.shape[0]),
        "vertexCount": int(result.vertices.shape[1]),
        "faceCount": int(result.faces.shape[0]),
        "verticesUrl": "vertices.f32",
        "facesUrl": "faces.i32",
        "audioUrl": "audio.wav",
        "motionsUrl": "motions.pt",
        "videoUrl": None,
        "avatarId": result.avatar_id,
    }
    write_web_metadata(metadata, output_dir)
    return metadata


def write_web_metadata(metadata, output_dir):
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)
