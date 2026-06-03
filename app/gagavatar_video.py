#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import importlib.util
import threading
from pathlib import Path

import torch

from app.avatar_registry import NEUTRAL_AVATAR_ID, get_tracked_avatar
from app.flame_model.FLAME import FLAMEModel


GAGAVATAR_RENDER_MODE = "gagavatar"
MESH_RENDER_MODE = "mesh"


def check_gagavatar_render_environment(device="auto"):
    if _select_render_device(device).type != "cuda":
        raise RuntimeError(
            "Server-side GAGAvatar rendering requires a CUDA device. "
            "Use mesh output on macOS, or run the backend on a CUDA server."
        )
    if importlib.util.find_spec("diff_gaussian_rasterization_32d") is None:
        raise RuntimeError(
            "Server-side GAGAvatar rendering requires diff_gaussian_rasterization_32d. "
            "Install the GAGAvatar gaussian rasterization dependency in the backend environment."
        )


class GAGAvatarVideoRenderer:
    def __init__(self, device="auto"):
        check_gagavatar_render_environment(device)
        self.device = _select_render_device(device)
        from app.GAGAvatar import GAGAvatar

        self.gagavatar = GAGAvatar().to(self.device)
        self.flame_model = FLAMEModel(
            n_shape=300,
            n_exp=100,
            scale=5.0,
            no_lmks=True,
        ).to(self.device)
        self._lock = threading.Lock()

    @torch.no_grad()
    def render_video(self, result, output_dir, *, avatar_id):
        if avatar_id in (None, "", NEUTRAL_AVATAR_ID):
            raise ValueError("Choose a GAGAvatar or uploaded avatar for colored video output.")
        output_dir = Path(output_dir)
        video_path = output_dir / "gagavatar.mp4"
        with self._lock:
            self.gagavatar.set_tracked_avatar(get_tracked_avatar(avatar_id), avatar_id)
            frames = []
            motions = result.motions.to(self.device)
            for motion in motions:
                batch = self.gagavatar.build_forward_batch(motion[None], self.flame_model)
                rgb = self.gagavatar.forward_expression(batch)
                frames.append(rgb.cpu()[0])
            video_frames = torch.stack(frames) * 255.0
            audio = result.audio[: int(video_frames.shape[0] / result.fps * result.sample_rate)]
            from app.utils_videos import write_video

            write_video(
                video_frames,
                str(video_path),
                float(result.fps),
                audio,
                result.sample_rate,
                "aac",
            )
        return video_path


_renderers = {}
_renderers_lock = threading.Lock()


def get_gagavatar_video_renderer(device="auto"):
    resolved = str(_select_render_device(device))
    with _renderers_lock:
        if resolved not in _renderers:
            _renderers[resolved] = GAGAvatarVideoRenderer(resolved)
        return _renderers[resolved]


def _select_render_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
