#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
from einops import repeat
from habitat.core.registry import registry

from .common import IdentityPE


@registry._register_impl(_type='model', to_register=None, name='PoseEmbedding')
class PoseEmbedding(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        img_feat_dim = cfg.MODEL.POSE_EMBED.IMG_FEAT_DIM
        hidden_dim = cfg.MODEL.POSE_EMBED.HIDDEN_DIM
        num_layers = cfg.MODEL.POSE_EMBED.NUM_LAYERS

        self.pos_encoder = registry._get_impl('pos_encoder', cfg.MODEL.POSE_EMBED.POS_ENCODER)(
            d_model=hidden_dim,
        )

        self.frame_encoder = registry._get_impl('image_encoder', cfg.MODEL.POSE_EMBED.FRAME_ENCODER)(
            img_feat_dim,
            hidden_dim,
        )

        self.pose_trf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.drop = nn.Dropout(0.2)

        num_classes = 5 + 8 # discrete
        self.b_fc1 = nn.Bilinear(hidden_dim, hidden_dim, num_classes, bias=False)
        self.b_fc2 = nn.Linear(hidden_dim, num_classes)

    def forward_frames(self, frames, batch):
        frames = self.frame_encoder(frames, batch) # (B, S, 256)
        return frames

    def forward_encoder(self, frames, batch):
        
        src_key_padding_mask = None
        if batch is not None and 'obs_mask' in batch:
            src_key_padding_mask = batch['obs_mask']

        pose_embeds = self.pose_trf(frames, src_key_padding_mask=src_key_padding_mask) # (B, S, 256)
        pose_embeds = self.drop(pose_embeds)
        return pose_embeds

    def forward(self, batch):

        frames = self.forward_frames(batch['rgb'], batch) # (B, S, 256)
        pose_embeds = self.forward_encoder(frames, batch)
        
        # arrange into SxS grid
        S = pose_embeds.shape[1]
        start_reps = repeat(pose_embeds, 'b s h -> b s r h', r=S).contiguous()
        end_reps = repeat(pose_embeds, 'b s h -> b r s h', r=S).contiguous()
        preds = self.b_fc1(start_reps, end_reps) + self.b_fc2(end_reps-start_reps)

        return preds # (B, S, S, 12+5)


@registry._register_impl(_type='model', to_register=None, name='EnvStatePredictor')
class EnvStatePredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.img_feat_dim = cfg.MODEL.IMG_FEAT_DIM
        self.pose_dim = cfg.MODEL.POSE_DIM

        # observation encoder
        self.frame_encoder = registry._get_impl('image_encoder', cfg.MODEL.FRAME_ENCODER)(
            self.img_feat_dim,
            self.hidden_dim,
        )  

        # timestep encoder
        self.pos_encoder = registry._get_impl('pos_encoder', cfg.MODEL.POS_ENCODER)(
            d_model=self.hidden_dim,
        )

        # trajectory encoder
        self.env_encoder = registry._get_impl('env_encoder', cfg.MODEL.ENV_ENCODER)(cfg)

        # trajectory decoder
        self.env_decoder = registry._get_impl('decoder', cfg.MODEL.DECODER)(cfg)

        # head
        heads = [registry._get_impl('head', head)(cfg) for head in cfg.MODEL.HEAD]
        self.heads = nn.ModuleList(heads)

        self.pose_embedder = PoseEmbedding(cfg)
        self.pose_fc = nn.Linear(
            cfg.MODEL.HIDDEN_DIM + cfg.MODEL.POSE_DIM,
            cfg.MODEL.HIDDEN_DIM
        )

        if hasattr(cfg.MODEL, 'POSE_MODEL_WEIGHTS') and len(cfg.MODEL.POSE_MODEL_WEIGHTS) > 0:
            weights = torch.load(cfg.MODEL.POSE_MODEL_WEIGHTS, map_location='cpu')['state_dict']
            weights = {k.replace('net.', ''):v for k, v in weights.items()}
            self.pose_embedder.load_state_dict(weights)
            print (f'Loaded pose model weights from {os.path.basename(cfg.MODEL.POSE_MODEL_WEIGHTS)}')

        self.pose_embedder.b_fc1 = None
        self.pose_embedder.b_fc2 = None
        print ('[WARN] Freezing pose encoder weights')
        for name, param in self.pose_embedder.named_parameters():
            param.requires_grad = False


        # load pretrained weights
        if hasattr(cfg.MODEL, 'WEIGHTS') and len(cfg.MODEL.WEIGHTS) != 0:
            self.load_pretrained_weights(cfg.MODEL.WEIGHTS)


    def load_pretrained_weights(self, pretrained_weights):
        weights = torch.load(pretrained_weights, map_location='cpu')['state_dict']
        weights = {k[len('net.'):]: v for k, v in weights.items()} # remove 'net.' prefix
        for k in list(weights.keys()):
            if k.startswith('heads.'):
                weights.pop(k)

        missing, unexpected = self.load_state_dict(weights, strict=False)
        print (f'Loaded weights from {pretrained_weights}')
        print (f'Missing: {missing} | Unexpected: {unexpected}')

    @torch.no_grad()
    def get_pose_embeddings(self, frames, query, frame_inds, query_inds, batch):
        pose_rgb = torch.cat([frames, query], 1)
        pose_inds = torch.cat([frame_inds, query_inds], 1)

        trf_args = None
        if 'obs_mask' in batch:
            B, T, _ = query.shape
            obs_mask = torch.cat([batch['obs_mask'], torch.zeros(B, T).type_as(batch['obs_mask'])], 1)
            trf_args = {'obs_mask': obs_mask}

        pose_embed = self.pose_embedder.forward_encoder(pose_rgb, trf_args) # (B, S+T, pdim)
        return pose_embed

    def encode_with_pose(self, frames, pose_embed, inds):
        encoded = self.pose_fc(torch.cat([frames, pose_embed], 2))
        encoded = self.pos_encoder(encoded, inds)    
        return encoded

    def env_forward(self, batch):
        
        B = batch['rgb'].shape[0]
        S = batch['mem_blocks'].shape[2]
        T = batch['query_rgb'].shape[1]

        # [!!] single mask for every T (that's ok!)
        enc_args, dec_args = {}, {} 
        if 'obs_mask' in batch:
            enc_args = {'src_key_padding_mask': batch['obs_mask']}
            dec_args = {'memory_key_padding_mask': batch['obs_mask']}

        # for pose_embed
        with torch.no_grad():
            pose_mem_frames =  self.pose_embedder.forward_frames(batch['rgb'], batch) # (B, Sp, 256)
            pose_query_frames =  self.pose_embedder.forward_frames(batch['query_rgb'], batch) # (B, T, 256)

        # for env_encoder
        mem_frames = self.frame_encoder(batch['rgb'], batch) # (B, Sp, 256)
        query_frames = self.frame_encoder(batch['query_rgb'], batch) # (B, T, 256)

        output = []
        query_encoded = []
        for t in range(T):
            block_inds = batch['mem_blocks'][:, t] # (B, S)

            # pose trf
            frames_t = pose_mem_frames[torch.arange(B), block_inds.t()].transpose(0, 1) # (B, S, 256)
            query_t = pose_query_frames[:, t].unsqueeze(1) # (B, 1, 256)
            frame_inds = batch['frame_inds'][:, t]
            query_inds = batch['query_inds'][:, t].unsqueeze(1)
            pose_embed = self.get_pose_embeddings(frames_t, query_t, frame_inds, query_inds, batch)

            # env trf
            frames_t = mem_frames[torch.arange(B), block_inds.t()].transpose(0, 1) # (B, S, 256)
            frame_inds = batch['frame_inds'][:, t]
            frames_t = self.encode_with_pose(frames_t, pose_embed[:, :S], frame_inds)

            query_t = query_frames[:, t].unsqueeze(1) # (B, 1, 256)
            query_inds = batch['query_inds'][:, t].unsqueeze(1)
            query_t = self.encode_with_pose(query_t, pose_embed[:, -1:], query_inds)

            # encoder + decoder
            mem_t = self.env_encoder(frames_t, {}, **enc_args) # (B, S, 256)
            out_t = self.env_decoder(query_t, mem_t, **dec_args) # (B, T=1, 256)
            output.append(out_t)

            query_encoded.append(query_t)
        
        output = torch.cat(output, 1) # (B, T, 256)
        query_encoded = torch.cat(query_encoded, 1) # (B, T, 256)
        
        return output, query_encoded

    def forward(self, batch):
        output, query_encoded = self.env_forward(batch)
        preds = []
        for head in self.heads:
            preds.append(head(query_encoded, output, batch))
        return preds


