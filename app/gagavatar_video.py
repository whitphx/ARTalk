#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import importlib.util
import os
import threading
from pathlib import Path

import torch

from app.avatar_registry import NEUTRAL_AVATAR_ID, get_tracked_avatar
from app.flame_model.FLAME import FLAMEModel


GAGAVATAR_RENDER_MODE = "gagavatar"
MESH_RENDER_MODE = "mesh"
BROWSER_GAUSSIAN_RENDER_MODE = "browser-gaussian"
UPSAMPLER_PREVIEW_FRAME_COUNT = 32
UPSAMPLER_PREVIEW_FRAME_COUNT_ENV = "ARTALK_UPSAMPLER_PREVIEW_FRAMES"
UPSAMPLER_PREVIEW_FRAME_STRIDE_ENV = "ARTALK_UPSAMPLER_PREVIEW_STRIDE"


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
        reference_video_path = output_dir / "gaussians.reference.mp4"
        with self._lock:
            self.gagavatar.set_tracked_avatar(get_tracked_avatar(avatar_id), avatar_id)
            self._reset_dynamic_avatar_state()
            first_batch = None
            first_gs_params = None
            head_frames = []
            transform_frames = []
            reference_frames = []
            motions = result.motions.to(self.device)
            upsampler_preview_indices = _upsampler_preview_frame_indices(int(motions.shape[0]))
            upsampler_preview_index_set = set(upsampler_preview_indices)
            upsampler_input_files = []
            upsampler_input_shape = None
            from app.GAGAvatar.utils_renderer import render_gaussian

            for frame_index, motion in enumerate(motions):
                batch = self.gagavatar.build_forward_batch(motion[None], self.flame_model)
                if first_batch is None:
                    first_batch = batch
                head_frames.append(batch["t_points"][0].detach().float().cpu())
                transform_frames.append(batch["t_transform"][0].detach().float().cpu())
                reference_frames.append(self.gagavatar.forward_expression(batch).cpu()[0])
                if frame_index in upsampler_preview_index_set:
                    gs_params_frame = self.gagavatar.forward_gaussians(batch)
                    if frame_index == 0:
                        first_gs_params = gs_params_frame
                    upsampler_input = (
                        render_gaussian(
                            gs_params=gs_params_frame,
                            cam_matrix=batch["t_transform"],
                            cam_params=self.gagavatar.cam_params,
                        )["images"][0].detach().to(torch.float16).cpu()
                    )
                    upsampler_input_shape = list(upsampler_input.shape)
                    upsampler_input_file = f"gaussians.upsampler_input_{len(upsampler_input_files):03d}.f16"
                    upsampler_input.numpy().tofile(output_dir / upsampler_input_file)
                    if not upsampler_input_files:
                        upsampler_input.numpy().tofile(output_dir / "gaussians.upsampler_input_first.f16")
                    upsampler_input_files.append(upsampler_input_file)
            if first_batch is None:
                raise ValueError("Cannot export Gaussian snapshot for an empty animation.")
            gs_params = first_gs_params if first_gs_params is not None else self.gagavatar.forward_gaussians(first_batch)
            if not upsampler_input_files or upsampler_input_shape is None:
                raise ValueError("Cannot export upsampler preview for an empty animation.")
            head_positions = torch.stack(head_frames)
            transforms = torch.stack(transform_frames)
            reference_video_frames = torch.stack(reference_frames) * 255.0
            snapshot = {
                "xyz": gs_params["xyz"][0].detach().float().cpu(),
                "colors": gs_params["colors"][0].detach().float().cpu(),
                "opacities": gs_params["opacities"][0].detach().float().cpu(),
                "scales": gs_params["scales"][0].detach().float().cpu(),
                "rotations": gs_params["rotations"][0].detach().float().cpu(),
            }
            for name, tensor in snapshot.items():
                tensor.numpy().astype("float32", copy=False).tofile(output_dir / f"gaussians.{name}.f32")
            from app.utils_videos import write_video

            write_video(reference_video_frames, str(reference_video_path), result.fps)
            head_positions.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.head_xyz.f32")
            transforms.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.transforms.f32")
        return {
            "gaussianCount": int(snapshot["xyz"].shape[0]),
            "gaussianFormat": "gagavatar-first-frame-f32-v1",
            "gaussianColorChannels": int(snapshot["colors"].shape[1]),
            "gaussianUpsamplerInput": {
                "dtype": "float16",
                "shape": upsampler_input_shape,
                "frameCount": len(upsampler_input_files),
                "frameIndices": upsampler_preview_indices,
            },
            "gaussianHeadCount": int(head_positions.shape[1]),
            "gaussianHeadFrameCount": int(head_positions.shape[0]),
            "gaussianTransformFrameCount": int(transforms.shape[0]),
            "gaussianUrls": {
                "xyz": "gaussians.xyz.f32",
                "headXyz": "gaussians.head_xyz.f32",
                "transforms": "gaussians.transforms.f32",
                "referenceVideo": "gaussians.reference.mp4",
                "upsamplerInputFirst": "gaussians.upsampler_input_first.f16",
                "upsamplerInputFrames": upsampler_input_files,
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


def _sample_frame_indices(frame_count, max_frames):
    if frame_count <= 0 or max_frames <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames == 1:
        return [0]
    return sorted({
        round(index * (frame_count - 1) / (max_frames - 1))
        for index in range(max_frames)
    })


def _upsampler_preview_frame_indices(frame_count):
    if frame_count <= 0:
        return []
    stride = _upsampler_preview_frame_stride()
    if stride is not None:
        return list(range(0, frame_count, stride))
    frame_count_limit = _upsampler_preview_frame_count(frame_count)
    return _sample_frame_indices(frame_count, frame_count_limit)


def _upsampler_preview_frame_count(frame_count):
    raw_value = os.environ.get(UPSAMPLER_PREVIEW_FRAME_COUNT_ENV)
    if raw_value is None:
        return UPSAMPLER_PREVIEW_FRAME_COUNT
    if raw_value.lower() == "all":
        return frame_count
    try:
        return max(1, int(raw_value))
    except ValueError:
        return UPSAMPLER_PREVIEW_FRAME_COUNT


def _upsampler_preview_frame_stride():
    raw_value = os.environ.get(UPSAMPLER_PREVIEW_FRAME_STRIDE_ENV)
    if raw_value is None:
        return None
    try:
        return max(1, int(raw_value))
    except ValueError:
        return None


def _cuda_diagnostics():
    return (
        f"torch={torch.__version__}, "
        f"torch_cuda={torch.version.cuda}, "
        f"cuda_available={torch.cuda.is_available()}, "
        f"device_count={torch.cuda.device_count()}"
    )
