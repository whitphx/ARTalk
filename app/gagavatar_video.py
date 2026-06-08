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
BROWSER_GAUSSIAN_RENDER_MODE = "browser-gaussian"


def check_gagavatar_render_environment(device="auto"):
    selected_device = _select_render_device(device)
    if selected_device.type != "cuda":
        raise RuntimeError(
            "Server-side GAGAvatar rendering requires a CUDA device. "
            "Use mesh output on macOS, or run the backend on a CUDA server. "
            f"CUDA diagnostics: {_cuda_diagnostics()}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Server-side GAGAvatar rendering requires CUDA, but CUDA is not "
            "available to this backend process. "
            f"CUDA diagnostics: {_cuda_diagnostics()}"
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
            self._reset_dynamic_avatar_state()
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
                result.fps,
                audio,
                result.sample_rate,
                "aac",
            )
        return video_path

    @torch.no_grad()
    def export_gaussian_snapshot(self, result, output_dir, *, avatar_id):
        if avatar_id in (None, "", NEUTRAL_AVATAR_ID):
            raise ValueError("Choose a GAGAvatar or uploaded avatar for browser Gaussian output.")
        output_dir = Path(output_dir)
        with self._lock:
            self.gagavatar.set_tracked_avatar(get_tracked_avatar(avatar_id), avatar_id)
            self._reset_dynamic_avatar_state()
            first_batch = None
            head_frames = []
            motions = result.motions.to(self.device)
            for motion in motions:
                batch = self.gagavatar.build_forward_batch(motion[None], self.flame_model)
                if first_batch is None:
                    first_batch = batch
                head_frames.append(batch["t_points"][0].detach().float().cpu())
            if first_batch is None:
                raise ValueError("Cannot export Gaussian snapshot for an empty animation.")
            gs_params = self.gagavatar.forward_gaussians(first_batch)
            head_positions = torch.stack(head_frames)
            snapshot = {
                "xyz": gs_params["xyz"][0].detach().float().cpu(),
                "colors": gs_params["colors"][0].detach().float().cpu(),
                "opacities": gs_params["opacities"][0].detach().float().cpu(),
                "scales": gs_params["scales"][0].detach().float().cpu(),
                "rotations": gs_params["rotations"][0].detach().float().cpu(),
            }
            for name, tensor in snapshot.items():
                tensor.numpy().astype("float32", copy=False).tofile(output_dir / f"gaussians.{name}.f32")
            head_positions.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.head_xyz.f32")
        return {
            "gaussianCount": int(snapshot["xyz"].shape[0]),
            "gaussianFormat": "gagavatar-first-frame-f32-v1",
            "gaussianColorChannels": int(snapshot["colors"].shape[1]),
            "gaussianHeadCount": int(head_positions.shape[1]),
            "gaussianHeadFrameCount": int(head_positions.shape[0]),
            "gaussianUrls": {
                "xyz": "gaussians.xyz.f32",
                "headXyz": "gaussians.head_xyz.f32",
                "colors": "gaussians.colors.f32",
                "opacities": "gaussians.opacities.f32",
                "scales": "gaussians.scales.f32",
                "rotations": "gaussians.rotations.f32",
            },
            "gaussianCamera": {
                "focalX": float(self.gagavatar.cam_params["focal_x"]),
                "focalY": float(self.gagavatar.cam_params["focal_y"]),
                "size": list(self.gagavatar.cam_params["size"]),
            },
        }

    def _reset_dynamic_avatar_state(self):
        if hasattr(self.gagavatar, "upper_points"):
            del self.gagavatar.upper_points


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


def _cuda_diagnostics():
    return (
        f"torch={torch.__version__}, "
        f"torch_cuda={torch.version.cuda}, "
        f"cuda_available={torch.cuda.is_available()}, "
        f"device_count={torch.cuda.device_count()}"
    )
