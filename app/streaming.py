#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Streaming wrapper around BitwiseARModel inference.

The released model processes audio in fixed 4-second / 100-frame chunks,
and chunk-to-chunk state (previous code bits and the rolling
attention-feature buffer) is the only thing that crosses chunk
boundaries. ARTalkStreamer exposes that chunked computation as a
stateful ``feed`` / ``finish`` API so callers can drive inference from
an audio source that arrives incrementally (e.g. a WebRTC microphone
track).

Streaming output is bit-exact with one-shot
``BitwiseARModel.inference`` for the same audio at the same chunk
boundaries; ``scripts/check_streaming_parity.py`` asserts this.

Post-processing performed by the engine (savgol smoothing, eye-channel
zeroing, fix_pose) is NOT applied here — those operate on full
sequences and are out of scope for Phase 1. Apply equivalent causal
post-processing one layer up.
"""

import math

import torch
import torch.nn.functional as F


SAMPLE_RATE = 16000
FPS = 25.0


class ARTalkStreamer:
    def __init__(self, model, style_motion=None):
        self.model = model
        self.patch_audio_length = int(model.patch_nums[-1] / FPS * SAMPLE_RATE)
        self.frames_per_chunk = model.patch_nums[-1]
        self.motion_dim = model.basic_vae.motion_dim
        self.set_style(style_motion)
        self.reset()

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def set_style(self, style_motion=None):
        model = self.model
        if style_motion is not None:
            if style_motion.dim() == 2:
                style_motion = style_motion[None]
            style_motion = style_motion.to(device=self.device, dtype=self.dtype)
            motion_style = model.style_encoder(style_motion).detach()
            motion_style_cond = model.style_cond_embed(motion_style)[:, None]
            motion_style_cond = motion_style_cond * 1.1 - model.null_style_cond * 0.1
        else:
            motion_style_cond = model.null_style_cond
        self._motion_style_cond = motion_style_cond

    @torch.no_grad()
    def reset(self):
        model = self.model
        prev_motion = torch.zeros(
            1, self.frames_per_chunk, self.motion_dim,
            dtype=self.dtype, device=self.device,
        )
        prev_code_bits, _ = model.basic_vae.quant_to_vqidx(prev_motion, this_motion=None)
        prev_vqfeat = model.basic_vae.vqidx_to_ms_vqfeat(prev_code_bits)
        prev_attn_feat = torch.cat(
            [self._motion_style_cond, model.vqfeat_embed(prev_vqfeat)], dim=1,
        ).repeat(1, model.prev_ratio, 1)
        self._prev_code_bits = prev_code_bits
        self._prev_attn_feat = prev_attn_feat
        self._audio_buffer = torch.zeros(0, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def feed(self, audio):
        if audio.dim() != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        audio = audio.to(device=self.device, dtype=self.dtype)
        self._audio_buffer = torch.cat([self._audio_buffer, audio])

        outputs = []
        while self._audio_buffer.shape[0] >= self.patch_audio_length:
            chunk = self._audio_buffer[: self.patch_audio_length]
            self._audio_buffer = self._audio_buffer[self.patch_audio_length:]
            outputs.append(self._step_chunk(chunk[None]))
        if outputs:
            return torch.cat(outputs, dim=0)
        return torch.zeros(0, self.motion_dim, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def finish(self):
        valid_samples = self._audio_buffer.shape[0]
        if valid_samples == 0:
            return torch.zeros(0, self.motion_dim, dtype=self.dtype, device=self.device)
        valid_frames = math.ceil(valid_samples / SAMPLE_RATE * FPS)
        pad = self.patch_audio_length - valid_samples
        chunk = torch.cat([
            self._audio_buffer,
            torch.zeros(pad, dtype=self.dtype, device=self.device),
        ])
        self._audio_buffer = self._audio_buffer.new_zeros(0)
        motion = self._step_chunk(chunk[None])
        return motion[:valid_frames]

    @torch.no_grad()
    def _step_chunk(self, audio_chunk):
        model = self.model
        lvl_pos_embed = model.lvl_embed(model.lvl_idx) + model.pos_embed
        prev_lvl_pos_embed = (
            model.lvl_embed(model.lvl_idx).repeat(1, model.prev_ratio, 1)
            + model.prev_pos_embed
        )

        split_audio_feat = model.audio_encoder(audio_chunk).permute(0, 2, 1)
        split_audio_feats = [
            F.interpolate(split_audio_feat, size=(pn), mode="area").permute(0, 2, 1)
            for pn in model.patch_nums
        ]
        split_audio_cond = torch.cat(split_audio_feats, dim=1).detach()

        next_ar_vqfeat = self._motion_style_cond
        pred_motion_bits = None
        for pidx, _pn in enumerate(model.patch_nums):
            patch_audio_cond = split_audio_cond[:, : sum(model.patch_nums[: pidx + 1])]
            patch_attn_bias = model.attn_bias_for_masking[
                :,
                :,
                : sum(model.patch_nums[: pidx + 1]),
                : sum(model.patch_nums[: pidx + 1]) + sum(model.patch_nums) * model.prev_ratio,
            ]
            attn_feat = next_ar_vqfeat + lvl_pos_embed[:, : next_ar_vqfeat.shape[1]]
            for bidx in range(model.attn_depth):
                attn_feat = model.attn_blocks[bidx](
                    attn_feat,
                    self._prev_attn_feat + prev_lvl_pos_embed,
                    patch_audio_cond,
                    attn_bias=patch_attn_bias,
                )
            pred_motion_logits = model.logits_head(
                model.cond_logits_head(attn_feat, patch_audio_cond)
            )
            pred_motion_bits = pred_motion_logits.view(
                pred_motion_logits.shape[0], pred_motion_logits.shape[1], -1, 2,
            ).argmax(dim=-1)
            if pidx < len(model.patch_nums) - 1:
                next_ar_vqfeat = model.basic_vae.vqidx_to_ar_vqfeat(pidx, pred_motion_bits)
                next_ar_vqfeat = torch.cat(
                    [self._motion_style_cond, model.vqfeat_embed(next_ar_vqfeat)], dim=1,
                )

        _, this_pred_motion = model.basic_vae.vqidx_to_motion(
            self._prev_code_bits, pred_motion_bits
        )

        new_prev_code_bits, _ = model.basic_vae.quant_to_vqidx(this_pred_motion, this_motion=None)
        new_prev_vqfeat = model.basic_vae.vqidx_to_ms_vqfeat(new_prev_code_bits).detach()
        this_prev_attn_feat = torch.cat(
            [self._motion_style_cond, model.vqfeat_embed(new_prev_vqfeat)], dim=1,
        )
        new_prev_attn_feat = torch.cat(
            [self._prev_attn_feat[:, this_prev_attn_feat.shape[1]:], this_prev_attn_feat], dim=1,
        )

        self._prev_code_bits = new_prev_code_bits
        self._prev_attn_feat = new_prev_attn_feat

        return this_pred_motion[0]
