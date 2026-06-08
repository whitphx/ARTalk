#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Server-side wrapper for GAGAvatar's single-image tracking flow.

The tracking command follows the image-tracking shape from upstream
`GAGAvatar/inference.py`:
https://github.com/xg-chu/GAGAvatar/blob/main/inference.py

GAGAvatar is MIT licensed. Its bundled `core/libs/GAGAvatar_track` tracker is
CC BY-NC 4.0, so production use needs separate license review.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


TRACKER_PREFLIGHT_SCRIPT = r"""
import sys

import torch
import torch._dynamo
import torchvision
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine
from core.libs.GAGAvatar_track.engines.human_matting import StyleMatteEngine

torch._dynamo.config.suppress_errors = True
TrackEngine(focal_length=12.0, device="cpu")
StyleMatteEngine(device="cpu")._init_models()
sys.stdout.write(
    "GAGAvatar tracker environment OK "
    f"(python={sys.executable}, cuda={torch.cuda.is_available()}, torchvision={torchvision.__version__})\n"
)
"""

TRACKER_SCRIPT = r"""
import sys
from pathlib import Path

import torch
import torch._dynamo
import torchvision
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine


def _can_track_on_cuda():
    try:
        _check_pytorch3d_cuda_rasterizer()
        return True
    except Exception:
        return False


def _check_pytorch3d_cuda_rasterizer():
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available (python={sys.executable}, torch={torch.__version__}, "
            f"torch_cuda={torch.version.cuda}, device_count={torch.cuda.device_count()})"
        )

    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes

    verts = torch.tensor(
        [[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[[0, 1, 2]]], device="cuda", dtype=torch.int64)
    rasterize_meshes(Meshes(verts=verts, faces=faces), image_size=4)


torch._dynamo.config.suppress_errors = True
image_path = sys.argv[1]
output_pt = Path(sys.argv[2])
output_preview = Path(sys.argv[3])
device = sys.argv[4]
if device == "auto":
    device = "cuda" if _can_track_on_cuda() else "cpu"
elif device.startswith("cuda"):
    try:
        _check_pytorch3d_cuda_rasterizer()
    except Exception as exc:
        raise RuntimeError(
            "GAGAvatar tracking was requested on CUDA, but the configured "
            "GAGAvatar Python environment cannot run PyTorch3D CUDA rasterization. "
            "Use device=auto/cpu, or install CUDA-enabled PyTorch and PyTorch3D "
            f"in GAGAVATAR_PYTHON. Details: {exc}"
        ) from exc

track_engine = TrackEngine(focal_length=12.0, device=device)
image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
tracked = track_engine.track_image([image], [image_path])
if tracked is None or image_path not in tracked:
    raise RuntimeError("No face was detected in the uploaded image")

feature_data = tracked[image_path]
output_pt.parent.mkdir(parents=True, exist_ok=True)
torch.save({"avatar": feature_data}, output_pt)
torchvision.utils.save_image(torch.tensor(feature_data["vis_image"]), output_preview)
"""


def check_tracker_environment():
    repo_path, python_executable = _tracker_runtime()
    env = _tracker_env(repo_path)
    result = subprocess.run(
        [python_executable, "-c", TRACKER_PREFLIGHT_SCRIPT],
        cwd=repo_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(_format_preflight_error(python_executable, details))
    return result.stdout.strip()


def track_uploaded_avatar(image_path, output_dir, *, device="auto"):
    repo_path, python_executable = _tracker_runtime()
    env = _tracker_env(repo_path)

    output_dir = Path(output_dir)
    input_path = output_dir / f"input{Path(image_path).suffix.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, input_path)
    input_path = input_path.resolve()
    tracked_path = (output_dir / "tracked.pt").resolve()
    preview_path = (output_dir / "preview.jpg").resolve()

    command = [
        python_executable,
        "-c",
        TRACKER_SCRIPT,
        str(input_path),
        str(tracked_path),
        str(preview_path),
        device,
    ]
    result = subprocess.run(
        command,
        cwd=repo_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or "GAGAvatar tracking failed")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "label": input_path.stem,
                "sourceImage": input_path.name,
            },
            f,
        )


def _tracker_runtime():
    repo_path_value = os.environ.get("GAGAVATAR_REPO")
    python_executable = os.environ.get("GAGAVATAR_PYTHON", sys.executable)
    if not repo_path_value:
        raise RuntimeError(
            "Set GAGAVATAR_REPO to a local GAGAvatar checkout before registering uploaded avatars"
        )
    repo_path = Path(repo_path_value).expanduser()
    if not repo_path.exists():
        raise RuntimeError(
            "Set GAGAVATAR_REPO to a local GAGAvatar checkout before registering uploaded avatars"
        )
    return repo_path, python_executable


def _tracker_env(repo_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(repo_path, env.get("PYTHONPATH"))
    # The upstream tracker pulls in face_alignment, which may use torch.compile.
    # On macOS that can route through Inductor's OpenMP C++ build and fail before
    # falling back, so this subprocess explicitly stays on eager execution.
    env["TORCHDYNAMO_DISABLE"] = "1"
    return env


def _format_preflight_error(python_executable, details):
    return (
        "GAGAvatar tracker environment is not ready. "
        f"Python: {python_executable}. "
        "Set GAGAVATAR_PYTHON to a dedicated GAGAvatar/GAGAvatar_track environment, "
        "or install the tracker dependencies in that environment. "
        f"Preflight error:\n{details}"
    )


def _prepend_path(path, existing):
    if existing:
        return f"{path}{os.pathsep}{existing}"
    return str(path)
