#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import json
import math
import os

import ipdb
import torch
import torch.nn as nn
from core.libs.flame_model import FLAMEModel
from core.models.modules import calc_mesh_loss
from einops import rearrange

from .bsq_quantizer import MultiScaleBSQ
from .transformer import TransformerDecoder, TransformerEncoder


class ARTalkCodec(nn.Module):
    def __init__(self, model_cfg, init_submodule=None, **kwargs):
        super().__init__()
        self.code_dim = model_cfg.V_CODE_DIM
        self.motion_dim = model_cfg.MOTION_DIM
        self.patch_nums = model_cfg.V_PATCH_NUMS
        self.stats_path = model_cfg.STATS_PATH
        # Per-frame codec: every frame is encoded, quantized and decoded on
        # its own (patch_nums must be [1]), so the codec is causal by
        # construction while training still sees whole windows for the
        # velocity and smoothness losses.
        self.frame_independent = bool(getattr(model_cfg, "FRAME_INDEPENDENT", False))
        if self.frame_independent and list(self.patch_nums) != [1]:
            raise ValueError("FRAME_INDEPENDENT requires V_PATCH_NUMS [1]")

        self.encoder = TransformerEncoder(
            inp_dim=self.motion_dim,
            hidden_dim=model_cfg.T_HIDDEN_DIM,
            code_dim=self.code_dim,
            depth=model_cfg.T_DEPTH,
            n_heads=model_cfg.T_NUM_HEADS,
        )
        self.quantize = MultiScaleBSQ(codebook_dim=self.code_dim, scale_schedule=self.patch_nums)
        self.decoder = TransformerDecoder(
            code_dim=self.code_dim,
            hidden_dim=model_cfg.T_HIDDEN_DIM,
            out_dim=self.motion_dim,
            depth=model_cfg.T_DEPTH,
            n_heads=model_cfg.T_NUM_HEADS,
        )

        # stat & render
        motion_stats = json.load(open(self.stats_path, "r"))
        motion_mean = torch.tensor(motion_stats["motion_mean"]).float()
        motion_std = torch.tensor(motion_stats["motion_std"]).float()
        if motion_mean.shape[0] > self.motion_dim:
            motion_mean = torch.cat([motion_mean[:104], motion_mean[106:108], motion_mean[109:111]])
            motion_std = torch.cat([motion_std[:104], motion_std[106:108], motion_std[109:111]])
        self.register_buffer("motion_mean", motion_mean)
        self.register_buffer("motion_std", motion_std)

        # for training parts
        if init_submodule:
            self.face_decoder = FLAMEModel(n_shape=300, n_exp=100)
            self.face_decoder.eval()

    def forward(self, batch, training=True):
        gt_motion_code = batch["motion_code"]
        if gt_motion_code.dim() == 4:
            gt_motion_code = rearrange(gt_motion_code, "b mb l c -> (b mb) l c")
        # encode
        enc_in = self.norm_with_stats(gt_motion_code)
        if self.frame_independent:
            enc_in = rearrange(enc_in, "b l c -> (b l) 1 c")
        enc_out = self.encoder(enc_in)
        # quantize
        vq_out, _, vq_loss = self.quantize(enc_out)
        # decode
        dec_out = self.decoder(vq_out)
        if self.frame_independent:
            dec_out = rearrange(dec_out, "(b l) 1 c -> b l c", b=gt_motion_code.shape[0])
        pred_motion_code = self.unnorm_with_stats(dec_out)
        return {
            "gt_motion_code": gt_motion_code,
            "pred_motion_code": pred_motion_code,
            "vq_loss": vq_loss.mean(),
        }

    @torch.inference_mode()
    def inference(self, motion_code, **kwargs):
        if motion_code.dim() == 4:
            motion_code = rearrange(motion_code, "b mb l c -> (b mb) l c")
        batch_size, code_len, code_dim = motion_code.shape
        if self.frame_independent:
            pred_motion_code = self.vqidx_to_motion(self.quant_to_vqidx(motion_code))
            return {"pred_motion_code": pred_motion_code, "gt_motion_code": motion_code}
        # motion batchs
        pad_len = math.ceil(code_len / self.patch_nums[-1]) * self.patch_nums[-1]
        pad_code = motion_code.new_zeros(batch_size, pad_len - code_len, code_dim)
        code_splits = torch.cat([motion_code, pad_code], dim=1)
        code_splits = code_splits.split(self.patch_nums[-1], dim=1)
        all_pred_code = []
        for code_one in code_splits:
            code_idx = self.quant_to_vqidx(code_one)
            pred_code = self.vqidx_to_motion(code_idx)
            all_pred_code.append(pred_code)
        pred_motion_code = torch.cat(all_pred_code, dim=1)[:, :code_len]
        results = {"pred_motion_code": pred_motion_code, "gt_motion_code": motion_code}
        return results

    def _calc_losses(self, train_results, _loss_kwargs):
        gt_motion_code = train_results["gt_motion_code"]
        pred_motion_code = train_results["pred_motion_code"]
        vq_loss = train_results["vq_loss"]
        gt_head_pose, pred_head_pose = gt_motion_code[..., 100:104], pred_motion_code[..., 100:104]
        # rec loss
        exp_loss = torch.nn.functional.l1_loss(pred_motion_code, gt_motion_code)
        # head pose loss
        pose_loss = torch.nn.functional.l1_loss(pred_head_pose, gt_head_pose)
        # vel&smooth loss
        gt_pose_vel = gt_head_pose[:, 1:] - gt_head_pose[:, :-1]
        pred_pose_vel = pred_head_pose[:, 1:] - pred_head_pose[:, :-1]
        head_vel_loss = torch.nn.functional.mse_loss(pred_pose_vel, gt_pose_vel)
        head_smooth_loss = torch.nn.functional.mse_loss(pred_pose_vel[:, 1:], pred_pose_vel[:, :-1])
        loss = {
            "vq_loss": vq_loss * _loss_kwargs.VQ_WEIGHT,
            "exp_loss": exp_loss * _loss_kwargs.EXP_WEIGHT,
            "pose_loss": pose_loss * _loss_kwargs.POSE_WEIGHT,
            "head_vel_loss": head_vel_loss * _loss_kwargs.HEAD_VEL_WEIGHT,
            "head_smooth_loss": head_smooth_loss * _loss_kwargs.HEAD_SMOOTH_WEIGHT,
        }
        # mesh&lips loss
        gt_verts = self.face_decoder.get_flame_verts(gt_motion_code, with_headpose=False)
        pred_verts = self.face_decoder.get_flame_verts(pred_motion_code, with_headpose=False)
        mesh_loss, lips_loss = calc_mesh_loss(pred_verts, gt_verts)
        # mesh vel&smooth loss
        gt_mesh_vel = gt_verts[:, 1:] - gt_verts[:, :-1]
        pred_mesh_vel = pred_verts[:, 1:] - pred_verts[:, :-1]
        mesh_vel_loss = torch.nn.functional.mse_loss(pred_mesh_vel, gt_mesh_vel)
        mesh_smooth_loss = torch.nn.functional.mse_loss(pred_mesh_vel[:, 1:], pred_mesh_vel[:, :-1])
        loss["mesh_loss"] = mesh_loss * _loss_kwargs.MESH_WEIGHT
        loss["lips_loss"] = lips_loss * _loss_kwargs.LIPS_WEIGHT
        loss["mesh_vel_loss"] = mesh_vel_loss * _loss_kwargs.MESH_VEL_WEIGHT
        loss["mesh_smooth_loss"] = mesh_smooth_loss * _loss_kwargs.MESH_SMOOTH_WEIGHT
        # print({k: "{:.4f}".format(v.item()) for k, v in loss.items()})
        return loss

    def configure_optimizers(self, config):
        learning_rate = config.LEARNING_RATE
        # print("Learning rate: {}".format(learning_rate))
        # params
        normal_params, decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)
        # optimizer
        optimizer = torch.optim.AdamW(
            [
                {"params": normal_params, "lr": learning_rate},
                {"params": decay_params, "lr": learning_rate * 0.1},
            ],
            lr=learning_rate,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_ITER,
        )
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.LR_DECAY_RATE,
            total_iters=(config.LR_DECAY_ITER - config.WARMUP_ITER),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.WARMUP_ITER],
        )
        return optimizer, scheduler

    def norm_with_stats(self, motion_code):
        normed_motion_code = (motion_code - self.motion_mean) / (self.motion_std + 1e-8)
        return normed_motion_code

    def unnorm_with_stats(self, motion_code):
        unnormed_motion_code = motion_code * self.motion_std + self.motion_mean
        return unnormed_motion_code

    @torch.no_grad()
    def quant_to_vqidx(self, motion_code):
        enc_in = self.norm_with_stats(motion_code)
        if self.frame_independent:
            enc_in = rearrange(enc_in, "b l c -> (b l) 1 c")
        enc_out = self.encoder(enc_in)
        _, motion_code_idx, _ = self.quantize(enc_out)
        if self.frame_independent:
            motion_code_idx = rearrange(motion_code_idx, "(b l) 1 d -> b l d", b=motion_code.shape[0])
        return motion_code_idx

    @torch.no_grad()
    def vqidx_to_motion(self, motion_code_idx):
        batch_size = motion_code_idx.shape[0]
        if self.frame_independent:
            motion_code_idx = rearrange(motion_code_idx, "b l d -> (b l) 1 d")
        vq_out = self.quantize.vqidx_to_feat(motion_code_idx)
        dec_out = self.decoder(vq_out)
        if self.frame_independent:
            dec_out = rearrange(dec_out, "(b l) 1 c -> b l c", b=batch_size)
        motion_code = self.unnorm_with_stats(dec_out)
        return motion_code

    @torch.no_grad()
    def quant_to_sum_feat(self, motion_code):
        enc_in = self.norm_with_stats(motion_code)
        enc_out = self.encoder(enc_in)
        _, motion_code_idx, _ = self.quantize(enc_out)
        vq_out = self.quantize.vqidx_to_feat(motion_code_idx)
        return vq_out

    # for training of ms-gen model
    @torch.no_grad()
    def flip_quant_to_feat(self, motion_code, flip_ratio, feat_style):
        assert feat_style in ["accum_next", "accum_curr", "local_next", "local_curr"]
        enc_in = self.norm_with_stats(motion_code)
        enc_out = self.encoder(enc_in)
        real_vqidx, flip_vqidx = self.quantize.flip_feat_to_vqidx(enc_out, flip_ratio)
        if feat_style == "accum_next":
            motion_feat = self.quantize.vqidx_to_accum_next_feat(flip_vqidx)
        elif feat_style == "local_next":
            motion_feat = self.quantize.vqidx_to_local_next_feat(flip_vqidx)
        elif feat_style == "accum_curr":
            motion_feat = self.quantize.vqidx_to_accum_curr_feat(flip_vqidx)
        elif feat_style == "local_curr":
            motion_feat = self.quantize.vqidx_to_local_curr_feat(flip_vqidx)
        else:
            raise NotImplementedError("feat_style not supported")
        return motion_feat.detach(), real_vqidx

    # for inference of ms-gen model
    @torch.no_grad()
    def vqidx_to_next_feat(self, pred_vqidx, pidx, feat_style):
        assert feat_style in ["accum_next", "local_next"]
        if feat_style == "accum_next":
            motion_feat = self.quantize.vqidx_to_accum_next_feat(pred_vqidx, pidx)
        elif feat_style == "local_next":
            motion_feat = self.quantize.vqidx_to_local_next_feat(pred_vqidx, pidx)
        else:
            raise NotImplementedError("feat_style not supported")
        return motion_feat.detach()

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.encoder.train()
            self.decoder.train()
            self.quantize.train()
            if hasattr(self, "face_decoder"):
                self.face_decoder.eval()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.quantize.eval()
        return self

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(state.keys()):
            if key.startswith("face_decoder"):
                state.pop(key)
        return state
