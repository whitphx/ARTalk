"""Frame-level generator that samples one codec token per audio frame.

The causal frame model's regression heads are replaced by a per-bit
logits head over the per-frame codec's 32-bit token; training uses the
same per-bit cross-entropy and inference the same top-p sampling as the
chunk generator. Sampling, not regression, is what keeps generated motion
from collapsing to the conditional mean (docs/frame-token-generation.md).
"""

import json
import math

import torch
import torch.nn as nn
from dataclasses import replace

from artalk_frame import CausalFrameModel
from artalk_frame.model import FrameModelState
from core.models import build_model
from core.models.artalk_gen.models import sample_idx_with_top_p_


class ARTalkFrameToken(CausalFrameModel):
    def __init__(self, model_cfg, init_submodule=True, **kwargs):
        with open(model_cfg.STATS_PATH) as fh:
            stats = json.load(fh)
        super().__init__(
            motion_dim=model_cfg.MOTION_DIM,
            expression_dim=model_cfg.EXPRESSION_DIM,
            sample_rate=model_cfg.SAMPLE_RATE,
            motion_fps=model_cfg.MOTION_FPS,
            audio_dim=model_cfg.AUDIO_DIM,
            style_dim=model_cfg.STYLE_DIM,
            motion_embed_dim=model_cfg.MOTION_EMBED_DIM,
            hidden_dim=model_cfg.HIDDEN_DIM,
            num_layers=model_cfg.NUM_LAYERS,
            dropout=model_cfg.DROPOUT,
            style_dropout=model_cfg.STYLE_DROPOUT,
            motion_mean=stats["motion_mean"],
            motion_std=stats["motion_std"],
        )
        self.audio_free_ratio = float(model_cfg.AUDIO_FREE)

        # Frozen per-frame codec; its weights ship inside this model's
        # checkpoint, so only training loads them from VAE_PATH.
        self.base_codec = build_model(model_cfg.VAE_CONFIG, init_submodule=False)
        if init_submodule:
            vae_ckpt = torch.load(model_cfg.VAE_CONFIG.VAE_PATH, map_location="cpu", weights_only=True)
            missing, unexpected = self.base_codec.load_state_dict(vae_ckpt["model"], strict=False)
            stray = [k for k in missing if not k.startswith("face_decoder.")]
            if stray or unexpected:
                raise RuntimeError(f"VAE checkpoint mismatch: missing={stray}, unexpected={unexpected}")
        if not self.base_codec.frame_independent:
            raise ValueError("ARTalkFrameToken needs a FRAME_INDEPENDENT codec")
        self.base_codec.eval()
        for param in self.base_codec.parameters():
            param.requires_grad = False
        self.code_dim = self.base_codec.code_dim

        # The regression heads become one head over the token's bits.
        del self.expression_head, self.pose_head
        head_input_dim = self.hidden_dim + int(model_cfg.AUDIO_DIM) + self.style_dim
        self.bits_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.code_dim * 2),
        )

    # ---- training: teacher-forced, whole sequence through the GRU at once
    def forward(self, batch, training=True):
        audio = batch["audio"]
        gt_motion = batch["motion_code"]
        batch_size, frame_count, _ = gt_motion.shape
        audio_frames = self.audio_to_frames(audio)
        if audio_frames.shape[1] != frame_count:
            raise ValueError("audio and motion frame counts differ")
        audio_features = self.audio_encoder(audio_frames)  # (B, L, A)
        if self.training and self.audio_free_ratio > 0:
            drop = torch.rand(batch_size, 1, 1, device=audio.device) < self.audio_free_ratio
            audio_features = torch.where(drop, torch.zeros_like(audio_features), audio_features)
        style = self.encode_style(batch.get("style_motion_code"), batch_size, device=audio.device)

        # Previous motion for frame t is the ground-truth frame t-1; the
        # window's real predecessor seeds t=0 when the loader provides it.
        prev = batch.get("prev_motion_code")
        seed = prev[:, -1:] if prev is not None else gt_motion.new_zeros(batch_size, 1, self.motion_dim)
        previous = self.normalize_motion(torch.cat([seed, gt_motion[:, :-1]], dim=1))

        style_seq = style[:, None].expand(-1, frame_count, -1)
        recurrent_input = self.recurrent_input(
            torch.cat([audio_features, style_seq, self.previous_motion_encoder(previous)], dim=-1)
        )
        recurrent_output, _ = self.recurrent(recurrent_input)
        logits = self.bits_head(torch.cat([recurrent_output, audio_features, style_seq], dim=-1))
        gt_bits = self.base_codec.quant_to_vqidx(gt_motion)  # (B, L, code_dim)
        return {
            "audio": audio,
            "gt_motion_code": gt_motion,
            "pred_motion_logits": logits,  # (B, L, code_dim * 2)
            "gt_motion_bits": gt_bits,
        }

    def _calc_losses(self, train_results, _loss_kwargs):
        logits = train_results["pred_motion_logits"]
        b, l, _ = logits.shape
        logits = logits.view(b, l, -1, 2)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 3, 1, 2), train_results["gt_motion_bits"].type(torch.long)
        )
        return {"ce_loss": ce_loss * _loss_kwargs.CE_WEIGHT}

    # ---- inference: one sampled token per frame, decoded and fed back
    def _step_encoded(self, audio_feature, state, tau=1.0, cfg=1.0, top_p=0.97):
        previous_feature = self.previous_motion_encoder(state.previous_motion)
        recurrent_input = self.recurrent_input(
            torch.cat([audio_feature, state.style, previous_feature], dim=-1)
        )
        recurrent_output, hidden = self.recurrent(recurrent_input.unsqueeze(1), state.hidden)
        logits = self.bits_head(torch.cat([recurrent_output[:, 0], audio_feature, state.style], dim=-1))
        logits = logits.view(logits.shape[0], 1, -1, 2) / tau
        if cfg > 1.0:
            # Unconditional branch: same state, silent audio.
            uncond_input = self.recurrent_input(
                torch.cat([torch.zeros_like(audio_feature), state.style, previous_feature], dim=-1)
            )
            uncond_output, _ = self.recurrent(uncond_input.unsqueeze(1), state.hidden)
            uncond_logits = self.bits_head(
                torch.cat([uncond_output[:, 0], torch.zeros_like(audio_feature), state.style], dim=-1)
            ).view(logits.shape) / tau
            logits = cfg * logits + (1 - cfg) * uncond_logits
        bits = sample_idx_with_top_p_(logits, top_p=top_p)  # (B, 1, code_dim)
        motion = self.base_codec.vqidx_to_motion(bits)[:, 0]  # (B, motion_dim)
        next_state = FrameModelState(
            hidden=hidden, previous_motion=self.normalize_motion(motion), style=state.style
        )
        return motion, next_state

    def step(self, audio_frame, state, **sampling):
        if audio_frame.dim() == 2:
            audio_frame = audio_frame.unsqueeze(1)
        audio_features = self.audio_encoder(audio_frame)
        if audio_features.shape[1] != 1:
            raise ValueError("step accepts exactly one audio frame")
        return self._step_encoded(audio_features[:, 0], state, **sampling)

    @torch.inference_mode()
    def inference(self, audio, style_motion_code=None, prev_motion_code=None, motion_code=None,
                  tau=1.0, cfg=1.0, **kwargs):
        audio_frames = self.audio_to_frames(audio, pad_end=True)
        batch_size, frame_count, _ = audio_frames.shape
        state = self.init_stream_state(batch_size, style_motion_code, prev_motion_code)
        audio_features = self.audio_encoder(audio_frames)
        frames = []
        for idx in range(frame_count):
            motion, state = self._step_encoded(audio_features[:, idx], state, tau=tau, cfg=cfg)
            frames.append(motion)
        pred = torch.stack(frames, dim=1)
        results = {"audio": audio, "pred_motion_code": pred}
        if motion_code is not None:
            n = min(pred.shape[1], motion_code.shape[1])
            results["pred_motion_code"] = pred[:, :n]
            results["gt_motion_code"] = motion_code[:, :n]
        return results

    # ---- housekeeping shared with the other training wrappers
    def configure_optimizers(self, config):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.WARMUP_ITER
        )
        main = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=config.LR_DECAY_RATE,
            total_iters=config.LR_DECAY_ITER - config.WARMUP_ITER,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[config.WARMUP_ITER]
        )
        return optimizer, scheduler

    def train(self, mode=True):
        super().train(mode)
        self.base_codec.eval()
        return self
