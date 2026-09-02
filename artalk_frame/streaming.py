"""Stateful audio buffering and checkpoint loading for frame generation."""

from pathlib import Path

import torch

from .model import CausalFrameModel


class FrameByFrameStreamer:
    """Feed arbitrary audio chunks and receive complete 25 fps motion frames."""

    def __init__(self, model, style_motion=None, previous_motion=None):
        self.model = model
        self._style_motion = style_motion
        self._previous_motion = previous_motion
        self.reset()

    @property
    def samples_per_frame(self):
        return self.model.samples_per_frame

    def reset(self):
        self._audio_buffer = self.model.motion_mean.new_empty(0)
        self._state = self.model.init_stream_state(
            batch_size=1,
            style_motion=self._style_motion,
            previous_motion=self._previous_motion,
        )

    @torch.inference_mode()
    def feed(self, audio_samples):
        if audio_samples.dim() != 1:
            raise ValueError("audio_samples must be one-dimensional")
        audio_samples = audio_samples.to(
            device=self.model.device,
            dtype=self.model.motion_mean.dtype,
        )
        self._audio_buffer = torch.cat([self._audio_buffer, audio_samples])
        frame_count = self._audio_buffer.shape[0] // self.samples_per_frame
        if frame_count == 0:
            return self.model.motion_mean.new_empty(0, self.model.motion_dim)
        consumed = frame_count * self.samples_per_frame
        audio_frames = self._audio_buffer[:consumed].reshape(
            frame_count, self.samples_per_frame
        )
        self._audio_buffer = self._audio_buffer[consumed:]
        predictions = []
        for audio_frame in audio_frames:
            motion, self._state = self.model.step(audio_frame[None], self._state)
            predictions.append(motion[0])
        return torch.stack(predictions, dim=0)

    @torch.inference_mode()
    def finish(self, pad_end=False):
        if not pad_end or self._audio_buffer.numel() == 0:
            self._audio_buffer = self._audio_buffer[:0]
            return self.model.motion_mean.new_empty(0, self.model.motion_dim)
        padding = self.samples_per_frame - self._audio_buffer.shape[0]
        tail = torch.nn.functional.pad(self._audio_buffer, (0, padding))
        self._audio_buffer = self._audio_buffer[:0]
        motion, self._state = self.model.step(tail[None], self._state)
        return motion


def _model_kwargs(model_cfg):
    keys = {
        "MOTION_DIM": "motion_dim",
        "EXPRESSION_DIM": "expression_dim",
        "SAMPLE_RATE": "sample_rate",
        "MOTION_FPS": "motion_fps",
        "AUDIO_DIM": "audio_dim",
        "STYLE_DIM": "style_dim",
        "MOTION_EMBED_DIM": "motion_embed_dim",
        "HIDDEN_DIM": "hidden_dim",
        "NUM_LAYERS": "num_layers",
        "DROPOUT": "dropout",
        "STYLE_DROPOUT": "style_dropout",
    }
    return {target: model_cfg[source] for source, target in keys.items() if source in model_cfg}


def load_frame_model(checkpoint_path, device="cpu"):
    """Load an exported training checkpoint without constructing FLAME."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("frame model checkpoint must contain a model state dict")
    try:
        model_cfg = checkpoint["meta_cfg"]["MODEL"]
    except (KeyError, TypeError) as error:
        raise ValueError("frame model checkpoint is missing MODEL metadata") from error
    loader = model_cfg.get("LOADER")
    if loader != "artalk_stream.ARTalkStream":
        raise ValueError(f"unsupported frame model loader: {loader!r}")
    model = CausalFrameModel(**_model_kwargs(model_cfg))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [key for key in missing if not key.startswith("face_decoder.")]
    unexpected = [key for key in unexpected if not key.startswith("face_decoder.")]
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match frame model (missing={missing}, unexpected={unexpected})"
        )
    return model.to(device).eval()
