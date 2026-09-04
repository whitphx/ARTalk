"""Streaming adapter for retrained short-chunk ARTalk generators.

Checkpoints from ``train_code`` are a different architecture from the
released runtime model (no state-dict overlap), so they cannot load into
``artalk.models.BitwiseARModel``. This module loads them through the
training code itself and exposes the feed/finish/reset streamer surface
of ``artalk.streaming.ARTalkStreamer``, so a retrained generator can
drive the same consumers.

The generator's own ``inference()`` carries one chunk of previous motion
and one chunk of style, but the 1 s recipe trains with three chunks of
previous context and a 100-frame style window
(``docs/chunk-size-retraining.md``). Driving ``inference()`` one chunk
per call lets this adapter maintain those trained context lengths
externally instead of inheriting the narrower defaults.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from core.libs.utils import ConfigDict
from core.models import build_model

PIPELINE_SAMPLE_RATE = 16_000
FPS = 25


def load_artalk1s_model(checkpoint_path: str | Path, device):
    """Build the training-code generator from a self-contained checkpoint.

    ``init_submodule=False`` skips the config's ``VAE_PATH`` reload: the
    codec weights ship inside the checkpoint. The audio encoder loads from
    the Hugging Face cache and is excluded from checkpoints by design.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "meta_cfg" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no meta_cfg; not a self-contained checkpoint")
    meta_cfg = ConfigDict(ckpt["meta_cfg"], gpus=1, cli_args=[])
    model = build_model(meta_cfg.MODEL, init_submodule=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    stray = [
        k for k in missing
        if not k.startswith(("audio_encoder.", "base_codec.face_decoder."))
    ]
    if stray or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={stray}, unexpected={unexpected}")
    model.eval().to(device)
    model._artalk1s_meta_cfg = meta_cfg
    return model


class ARTalk1sStreamer:
    """Drives a train-code ARTalk generator chunk-by-chunk over live audio."""

    def __init__(self, model, tau: float = 1.0, cfg: float = 2.0):
        self.model = model
        self.tau = float(tau)
        self.cfg = float(cfg)
        self.frames_per_chunk = max(model.patch_nums)
        self.patch_audio_length = int(self.frames_per_chunk / FPS * PIPELINE_SAMPLE_RATE)
        self.motion_dim = model.motion_dim
        meta = model._artalk1s_meta_cfg
        self._prev_frames = int(meta.DATASET.PREV_LENGTH)
        self._style_frames = int(meta.DATASET.STYLE_LENGTH)
        self.reset()

    @property
    def device(self):
        return self.model.device

    @torch.inference_mode()
    def reset(self):
        self._audio_buffer = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._prev_motion = torch.zeros(
            1, self._prev_frames, self.motion_dim, dtype=torch.float32, device=self.device
        )
        # Zero style is the recipe's unconditional-style input (STYLE_FREE
        # training drops style the same way).
        self._style_motion = torch.zeros(
            1, self._style_frames, self.motion_dim, dtype=torch.float32, device=self.device
        )

    @torch.inference_mode()
    def feed(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        audio = audio.to(device=self.device, dtype=torch.float32)
        self._audio_buffer = torch.cat([self._audio_buffer, audio])

        outputs = []
        while self._audio_buffer.shape[0] >= self.patch_audio_length:
            chunk = self._audio_buffer[: self.patch_audio_length]
            self._audio_buffer = self._audio_buffer[self.patch_audio_length:]
            outputs.append(self._step_chunk(chunk))
        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.zeros(0, self.motion_dim, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def finish(self) -> torch.Tensor:
        valid_samples = self._audio_buffer.shape[0]
        if valid_samples == 0:
            return torch.zeros(0, self.motion_dim, dtype=torch.float32, device=self.device)
        valid_frames = math.ceil(valid_samples / PIPELINE_SAMPLE_RATE * FPS)
        pad = self.patch_audio_length - valid_samples
        chunk = torch.cat([self._audio_buffer, self._audio_buffer.new_zeros(pad)])
        self._audio_buffer = self._audio_buffer.new_zeros(0)
        return self._step_chunk(chunk)[:valid_frames]

    def _step_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        out = self.model.inference(
            chunk[None],
            style_motion_code=self._style_motion,
            prev_motion_code=self._prev_motion,
            tau=self.tau,
            cfg=self.cfg,
        )
        motion = out["pred_motion_code"]  # (1, frames_per_chunk, motion_dim)
        self._prev_motion = torch.cat([self._prev_motion, motion], dim=1)[
            :, -self._prev_frames :
        ]
        return motion[0]
