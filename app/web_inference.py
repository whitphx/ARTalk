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
from scipy.io import wavfile
from scipy.signal import savgol_filter

from app.avatar_registry import get_avatar_shape_code
from app.models import BitwiseARModel
from app.flame_model.FLAME import FLAMEModel


STYLE_DIR = Path("assets/style_motion")
MESH_REGION_LABELS = {
    "skin": 0,
    "lips": 1,
    "mouth": 2,
    "eye": 3,
}
MEDIAPIPE_EYE_LANDMARKS = {
    7,
    33,
    133,
    144,
    145,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    246,
    249,
    263,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    390,
    398,
    466,
}
MEDIAPIPE_LIP_LANDMARKS = {
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
}
MEDIAPIPE_MOUTH_LANDMARKS = {
    13,
    14,
    17,
    78,
    80,
    81,
    82,
    87,
    88,
    95,
    178,
    191,
    308,
    310,
    311,
    312,
    317,
    318,
    324,
    402,
    415,
}


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


def load_audio(audio_path):
    path = Path(audio_path)
    try:
        return torchaudio.load(str(path))
    except RuntimeError as exc:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                "Could not decode audio with torchaudio. Install an audio backend "
                "such as ffmpeg/soundfile in the web backend environment, or upload a WAV file."
            ) from exc
        try:
            sr, data = wavfile.read(path)
        except Exception as wav_exc:
            raise exc from wav_exc
        if data.ndim == 1:
            data = data[None, :]
        else:
            data = data.T
        audio_np = data.astype(np.float32, copy=False)
        if np.issubdtype(data.dtype, np.unsignedinteger):
            info = np.iinfo(data.dtype)
            midpoint = float(info.max + 1) / 2.0
            audio_np = (audio_np - midpoint) / midpoint
        elif np.issubdtype(data.dtype, np.signedinteger):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), info.max))
            audio_np = audio_np / scale
        return torch.from_numpy(audio_np), sr


def save_audio(audio_path, audio, sample_rate):
    path = Path(audio_path)
    try:
        torchaudio.save(str(path), audio, sample_rate)
        return
    except RuntimeError as exc:
        if path.suffix.lower() != ".wav":
            raise exc
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    audio_np = np.clip(audio_np, -1.0, 1.0)
    wavfile.write(path, sample_rate, (audio_np * 32767.0).astype(np.int16))


def build_mesh_region_labels(vertex_count, faces, region_seed_faces):
    labels = np.full(vertex_count, MESH_REGION_LABELS["skin"], dtype=np.uint8)
    neighbors = build_vertex_neighbors(vertex_count, faces)

    for name, depth in (("lips", 2), ("eye", 1), ("mouth", 1)):
        seed_faces = region_seed_faces.get(name)
        if seed_faces is None or len(seed_faces) == 0:
            continue
        vertices = grow_region_vertices(faces[seed_faces].reshape(-1), neighbors, depth)
        labels[vertices] = MESH_REGION_LABELS[name]
    return labels


def build_vertex_neighbors(vertex_count, faces):
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def grow_region_vertices(seed_vertices, neighbors, depth):
    region = set(int(vertex) for vertex in seed_vertices)
    frontier = set(region)
    for _ in range(depth):
        next_frontier = set()
        for vertex in frontier:
            next_frontier.update(neighbors[vertex])
        next_frontier.difference_update(region)
        region.update(next_frontier)
        frontier = next_frontier
    return np.fromiter(region, dtype=np.int64)


def mediapipe_region_seed_faces(flame_model):
    ckpt = flame_model.flame_ckpt["lmk_embeddings_mediapipe"]
    landmark_ids = ckpt["landmark_indices"].detach().cpu().numpy()
    landmark_face_indices = flame_model.lmk_faces_idx_mediapipe.detach().cpu().numpy()
    return {
        "lips": landmark_faces_for_ids(landmark_ids, landmark_face_indices, MEDIAPIPE_LIP_LANDMARKS),
        "mouth": landmark_faces_for_ids(landmark_ids, landmark_face_indices, MEDIAPIPE_MOUTH_LANDMARKS),
        "eye": landmark_faces_for_ids(landmark_ids, landmark_face_indices, MEDIAPIPE_EYE_LANDMARKS),
    }


def landmark_faces_for_ids(landmark_ids, landmark_face_indices, selected_ids):
    mask = np.isin(landmark_ids, list(selected_ids))
    return np.unique(landmark_face_indices[mask]).astype(np.int64, copy=False)


@dataclass
class WebInferenceResult:
    audio: torch.Tensor
    motions: torch.Tensor
    vertices: np.ndarray
    faces: np.ndarray
    region_labels: np.ndarray
    region_source: str
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
            audio, sr = load_audio(audio_path)
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
            faces = self.flame_model.get_faces().cpu().numpy().astype(np.int32, copy=False)
            return WebInferenceResult(
                audio=audio.float().cpu(),
                motions=pred_motions.float().cpu(),
                vertices=vertices.float().cpu().numpy().astype(np.float32, copy=False),
                faces=faces,
                region_labels=build_mesh_region_labels(
                    int(vertices.shape[1]),
                    faces,
                    mediapipe_region_seed_faces(self.flame_model),
                ),
                region_source="mediapipe-landmark-adjacency-v1",
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
    result.region_labels.tofile(output_dir / "regions.u8")
    torch.save(result.motions, output_dir / "motions.pt")
    save_audio(output_dir / "audio.wav", result.audio[None], result.sample_rate)
    metadata = {
        "renderMode": "mesh",
        "fps": result.fps,
        "sampleRate": result.sample_rate,
        "frameCount": int(result.vertices.shape[0]),
        "vertexCount": int(result.vertices.shape[1]),
        "faceCount": int(result.faces.shape[0]),
        "verticesUrl": "vertices.f32",
        "facesUrl": "faces.i32",
        "regionLabelsUrl": "regions.u8",
        "regionLabelFormat": "uint8-vertex",
        "regionLabels": MESH_REGION_LABELS,
        "regionSource": result.region_source,
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
