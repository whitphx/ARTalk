"""Shared causal model used by both training and realtime inference."""

from dataclasses import dataclass, replace

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MotionLayout:
    """Named boundary between expression and pose motion parameters."""

    motion_dim: int
    expression_dim: int = 100

    def __post_init__(self):
        if self.motion_dim <= 0:
            raise ValueError("motion_dim must be positive")
        if not 0 < self.expression_dim < self.motion_dim:
            raise ValueError("expression_dim must be between 0 and motion_dim")

    @property
    def pose_dim(self):
        return self.motion_dim - self.expression_dim

    @property
    def expression_slice(self):
        return slice(0, self.expression_dim)

    @property
    def pose_slice(self):
        return slice(self.expression_dim, self.motion_dim)

    def validate(self, motion, name="motion"):
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(
                f"{name} must have {self.motion_dim} parameters, got {motion.shape[-1]}"
            )


@dataclass(frozen=True)
class FrameModelState:
    """Bounded state carried between emitted motion frames."""

    hidden: torch.Tensor
    previous_motion: torch.Tensor
    style: torch.Tensor


class AudioFrameEncoder(nn.Module):
    """Encode one complete 40 ms audio frame without future context."""

    def __init__(self, samples_per_frame, output_dim):
        super().__init__()
        self.samples_per_frame = int(samples_per_frame)
        self.network = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=5),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv1d(128, 192, kernel_size=4, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(192, output_dim)

    def forward(self, audio_frames):
        if audio_frames.dim() != 3:
            raise ValueError("audio_frames must have shape [batch, frames, samples]")
        if audio_frames.shape[-1] != self.samples_per_frame:
            raise ValueError(
                f"each audio frame must contain {self.samples_per_frame} samples"
            )
        batch_size, frame_count, _ = audio_frames.shape
        audio = audio_frames.reshape(batch_size * frame_count, 1, self.samples_per_frame)
        mean = audio.mean(dim=-1, keepdim=True)
        variance = audio.var(dim=-1, keepdim=True, unbiased=False)
        audio = (audio - mean) * torch.rsqrt(variance + 1e-5)
        features = self.network(audio).flatten(1)
        features = self.projection(features)
        return features.reshape(batch_size, frame_count, -1)


class MotionStyleEncoder(nn.Module):
    def __init__(self, motion_dim, style_dim):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(motion_dim),
            nn.Linear(motion_dim, style_dim),
            nn.GELU(),
            nn.Linear(style_dim, style_dim),
        )
        self.output_norm = nn.LayerNorm(style_dim)

    def forward(self, normalized_motion):
        return self.output_norm(self.frame_encoder(normalized_motion).mean(dim=1))


class CausalFrameModel(nn.Module):
    """Generate exactly one motion frame from each 40 ms audio frame.

    A GRU provides constant-cost recurrent state. Audio framing is explicit,
    which prevents training from accidentally using samples from future frames.
    """

    def __init__(
        self,
        motion_dim=108,
        expression_dim=100,
        sample_rate=16000,
        motion_fps=25,
        audio_dim=192,
        style_dim=128,
        motion_embed_dim=128,
        hidden_dim=384,
        num_layers=2,
        dropout=0.1,
        style_dropout=0.1,
        motion_mean=None,
        motion_std=None,
    ):
        super().__init__()
        if sample_rate % motion_fps != 0:
            raise ValueError("sample_rate must be divisible by motion_fps")
        self.layout = MotionLayout(int(motion_dim), int(expression_dim))
        self.motion_dim = self.layout.motion_dim
        self.sample_rate = int(sample_rate)
        self.motion_fps = int(motion_fps)
        self.samples_per_frame = self.sample_rate // self.motion_fps
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.style_dim = int(style_dim)
        self.style_dropout = float(style_dropout)
        if not 0.0 <= self.style_dropout <= 1.0:
            raise ValueError("style_dropout must be between 0 and 1")

        if motion_mean is None:
            motion_mean = torch.zeros(self.motion_dim)
        if motion_std is None:
            motion_std = torch.ones(self.motion_dim)
        motion_mean = torch.as_tensor(motion_mean, dtype=torch.float32)
        motion_std = torch.as_tensor(motion_std, dtype=torch.float32)
        if motion_mean.shape != (self.motion_dim,) or motion_std.shape != (self.motion_dim,):
            raise ValueError("motion statistics must match motion_dim")
        if torch.any(motion_std <= 0):
            raise ValueError("motion_std must be strictly positive")
        self.register_buffer("motion_mean", motion_mean)
        self.register_buffer("motion_std", motion_std)

        self.audio_encoder = AudioFrameEncoder(self.samples_per_frame, audio_dim)
        self.style_encoder = MotionStyleEncoder(self.motion_dim, style_dim)
        self.null_style = nn.Parameter(torch.zeros(1, style_dim))
        self.previous_motion_encoder = nn.Sequential(
            nn.LayerNorm(self.motion_dim),
            nn.Linear(self.motion_dim, motion_embed_dim),
            nn.GELU(),
        )
        recurrent_input_dim = audio_dim + style_dim + motion_embed_dim
        self.recurrent_input = nn.Sequential(
            nn.Linear(recurrent_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        head_input_dim = hidden_dim + audio_dim + style_dim
        self.expression_head = self._build_head(
            head_input_dim, hidden_dim, self.layout.expression_dim
        )
        self.pose_head = self._build_head(
            head_input_dim, max(hidden_dim // 2, 64), self.layout.pose_dim
        )

    @staticmethod
    def _build_head(input_dim, hidden_dim, output_dim):
        head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @property
    def device(self):
        return self.motion_mean.device

    def normalize_motion(self, motion):
        self.layout.validate(motion)
        return (motion - self.motion_mean) / self.motion_std

    def denormalize_motion(self, motion):
        self.layout.validate(motion, "normalized motion")
        return motion * self.motion_std + self.motion_mean

    def audio_to_frames(self, audio, pad_end=False):
        if audio.dim() == 3:
            if audio.shape[-1] != self.samples_per_frame:
                raise ValueError("pre-framed audio has an invalid frame size")
            return audio
        if audio.dim() != 2:
            raise ValueError("audio must have shape [batch, samples] or [batch, frames, samples]")
        remainder = audio.shape[-1] % self.samples_per_frame
        if remainder:
            if pad_end:
                audio = torch.nn.functional.pad(audio, (0, self.samples_per_frame - remainder))
            else:
                audio = audio[:, : audio.shape[-1] - remainder]
        if audio.shape[-1] == 0:
            return audio.new_empty(audio.shape[0], 0, self.samples_per_frame)
        return audio.unflatten(-1, (-1, self.samples_per_frame))

    def encode_style(self, style_motion, batch_size, device=None, dtype=None):
        device = self.device if device is None else device
        dtype = self.motion_mean.dtype if dtype is None else dtype
        if style_motion is None:
            return self.null_style.to(device=device, dtype=dtype).expand(batch_size, -1)
        style_motion = style_motion.to(device=device, dtype=dtype)
        if style_motion.dim() == 2:
            style_motion = style_motion.unsqueeze(0)
        if style_motion.shape[0] == 1 and batch_size > 1:
            style_motion = style_motion.expand(batch_size, -1, -1)
        if style_motion.shape[0] != batch_size:
            raise ValueError("style batch size must match audio batch size")
        style = self.style_encoder(self.normalize_motion(style_motion))
        if self.training and self.style_dropout > 0:
            drop_mask = torch.rand(batch_size, 1, device=device) < self.style_dropout
            null_style = self.null_style.to(device=device, dtype=dtype).expand_as(style)
            style = torch.where(drop_mask, null_style, style)
        return style

    def init_stream_state(self, batch_size=1, style_motion=None, previous_motion=None):
        if previous_motion is None:
            normalized_previous = self.motion_mean.new_zeros(batch_size, self.motion_dim)
        else:
            if previous_motion.dim() == 3:
                previous_motion = previous_motion[:, -1]
            previous_motion = previous_motion.to(
                device=self.device, dtype=self.motion_mean.dtype
            )
            if previous_motion.shape != (batch_size, self.motion_dim):
                raise ValueError("previous_motion must have shape [batch, motion_dim]")
            normalized_previous = self.normalize_motion(previous_motion)
        hidden = self.motion_mean.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        style = self.encode_style(
            style_motion,
            batch_size,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return FrameModelState(hidden=hidden, previous_motion=normalized_previous, style=style)

    def _step_encoded(self, audio_feature, state):
        previous_feature = self.previous_motion_encoder(state.previous_motion)
        recurrent_input = self.recurrent_input(
            torch.cat([audio_feature, state.style, previous_feature], dim=-1)
        )
        recurrent_output, hidden = self.recurrent(
            recurrent_input.unsqueeze(1), state.hidden
        )
        head_input = torch.cat(
            [recurrent_output[:, 0], audio_feature, state.style], dim=-1
        )
        normalized_motion = torch.cat(
            [self.expression_head(head_input), self.pose_head(head_input)], dim=-1
        )
        motion = self.denormalize_motion(normalized_motion)
        next_state = FrameModelState(
            hidden=hidden,
            previous_motion=normalized_motion,
            style=state.style,
        )
        return motion, next_state

    def step(self, audio_frame, state):
        if audio_frame.dim() == 2:
            audio_frame = audio_frame.unsqueeze(1)
        audio_features = self.audio_encoder(audio_frame)
        if audio_features.shape[1] != 1:
            raise ValueError("step accepts exactly one audio frame")
        return self._step_encoded(audio_features[:, 0], state)

    def generate(
        self,
        audio,
        style_motion=None,
        previous_motion=None,
        teacher_motion=None,
        teacher_forcing_ratio=0.0,
        pad_end=False,
    ):
        audio_frames = self.audio_to_frames(audio, pad_end=pad_end)
        batch_size, frame_count, _ = audio_frames.shape
        if teacher_motion is not None:
            self.layout.validate(teacher_motion, "teacher_motion")
            if teacher_motion.shape[:2] != (batch_size, frame_count):
                raise ValueError("teacher_motion must align with framed audio")
        state = self.init_stream_state(batch_size, style_motion, previous_motion)
        audio_features = self.audio_encoder(audio_frames)
        predictions = []
        for frame_idx in range(frame_count):
            prediction, state = self._step_encoded(audio_features[:, frame_idx], state)
            predictions.append(prediction)
            if teacher_motion is not None and teacher_forcing_ratio > 0:
                teacher_mask = torch.rand(
                    batch_size, 1, device=prediction.device
                ) < teacher_forcing_ratio
                normalized_teacher = self.normalize_motion(teacher_motion[:, frame_idx])
                next_previous = torch.where(
                    teacher_mask, normalized_teacher, state.previous_motion
                )
                state = replace(state, previous_motion=next_previous)
        if not predictions:
            return audio.new_empty(batch_size, 0, self.motion_dim)
        return torch.stack(predictions, dim=1)

    def forward(
        self,
        audio,
        style_motion=None,
        previous_motion=None,
        teacher_motion=None,
        teacher_forcing_ratio=0.0,
    ):
        return self.generate(
            audio,
            style_motion=style_motion,
            previous_motion=previous_motion,
            teacher_motion=teacher_motion,
            teacher_forcing_ratio=teacher_forcing_ratio,
            pad_end=False,
        )
