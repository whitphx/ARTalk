#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


BUILTIN_TRACKED_PATH = Path("assets/GAGAvatar/tracked.pt")
UPLOADED_AVATAR_ROOT = Path("render_results/web_avatars")
NEUTRAL_AVATAR_ID = "mesh"


@dataclass(frozen=True)
class AvatarInfo:
    id: str
    label: str
    source: str
    previewUrl: str | None = None


def available_avatars():
    avatars = [
        AvatarInfo(
            id=NEUTRAL_AVATAR_ID,
            label="Neutral mesh",
            source="mesh",
        )
    ]
    avatars.extend(_builtin_avatars())
    avatars.extend(_uploaded_avatars())
    return [asdict(avatar) for avatar in avatars]


def get_avatar_shape_code(avatar_id):
    if avatar_id in (None, "", NEUTRAL_AVATAR_ID):
        return None
    tracked = get_tracked_avatar(avatar_id)
    shapecode = torch.as_tensor(tracked["shapecode"], dtype=torch.float32)
    if tuple(shapecode.shape) != (300,):
        raise ValueError(f"Invalid shapecode shape for {avatar_id}: {tuple(shapecode.shape)}")
    return shapecode[None]


def get_tracked_avatar(avatar_id):
    source, key = _split_avatar_id(avatar_id)
    if source == "gagavatar":
        tracked = _load_tracked_file(BUILTIN_TRACKED_PATH)
        if key not in tracked:
            raise KeyError(f"Unknown built-in avatar: {avatar_id}")
        return tracked[key]
    if source == "uploaded":
        avatar_dir = _uploaded_avatar_dir(key)
        tracked = _load_tracked_file(avatar_dir / "tracked.pt")
        if "avatar" not in tracked:
            raise KeyError(f"Uploaded avatar is missing tracking data: {avatar_id}")
        return tracked["avatar"]
    raise KeyError(f"Unknown avatar: {avatar_id}")


def _builtin_avatars():
    if not BUILTIN_TRACKED_PATH.exists():
        return []
    tracked = _load_tracked_file(BUILTIN_TRACKED_PATH)
    avatars = []
    for key in sorted(tracked):
        avatars.append(
            AvatarInfo(
                id=f"gagavatar:{key}",
                label=f"GAGAvatar {Path(key).stem}",
                source="gagavatar",
            )
        )
    return avatars


def _uploaded_avatars():
    if not UPLOADED_AVATAR_ROOT.exists():
        return []
    avatars = []
    for avatar_dir in sorted(UPLOADED_AVATAR_ROOT.iterdir()):
        if not avatar_dir.is_dir() or not (avatar_dir / "tracked.pt").exists():
            continue
        metadata = _read_metadata(avatar_dir / "metadata.json")
        avatar_id = avatar_dir.name
        avatars.append(
            AvatarInfo(
                id=f"uploaded:{avatar_id}",
                label=metadata.get("label") or f"Uploaded {avatar_id[:8]}",
                source="uploaded",
                previewUrl=f"/api/avatar-jobs/{avatar_id}/preview.jpg",
            )
        )
    return avatars


def _split_avatar_id(avatar_id):
    if ":" not in avatar_id:
        raise KeyError(f"Unknown avatar: {avatar_id}")
    source, key = avatar_id.split(":", 1)
    if not key:
        raise KeyError(f"Unknown avatar: {avatar_id}")
    return source, key


def _uploaded_avatar_dir(avatar_id):
    if not avatar_id or any(char not in "0123456789abcdef" for char in avatar_id):
        raise KeyError(f"Unknown uploaded avatar: {avatar_id}")
    return UPLOADED_AVATAR_ROOT / avatar_id


def _read_metadata(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_tracked_file(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)
