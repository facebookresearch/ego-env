#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math
from einops import rearrange

from habitat.core.registry import registry

# ------------------------------------------------------------------------------------------ #
# Input: (B, N, hidden_size)
# Output: (B, N, hidden_size)
@registry._register_impl(_type='pos_encoder', to_register=None, name='Identity')
class IdentityPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512, **kwargs):
        super().__init__()
    def forward(self, x, inds):
        return x

# Sine/Cosine PE
@registry._register_impl(_type='pos_encoder', to_register=None, name='SinCos')
class SinCosPE(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512, **kwargs):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, inds):
        x = x + self.pe[inds]
        return x

# ------------------------------------------------------------------------------------------ #
# Input: (B, N, img_feat_dim)
# Output: (B, N, hidden_size)
@registry._register_impl(_type='image_encoder', to_register=None, name='Identity')
class IdentityFeatEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, **kwargs):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
    def forward(self, x, batch):
        x = self.drop(x)
        return x
    
@registry._register_impl(_type='image_encoder', to_register=None, name='FeatEncoder')
class FeatEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x, batch):
        return self.drop(self.mlp(x))

# ------------------------------------------------------------------------------------------ #
# Input: (B, S, hidden_size)
# Output: (B, S, hidden_size)
@registry._register_impl(_type='env_encoder', to_register=None, name='EnvEncoder')
class EnvEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.MODEL.ENCODER_LAYERS
        hidden_dim = cfg.MODEL.HIDDEN_DIM

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(self, x, batch, **kwargs): # x = (B, S, hidden_size)
        x = self.transformer_encoder(x, **kwargs) # (B, S, hidden_size)
        return x

# ------------------------------------------------------------------------------------------ #
# Input: memory (B, S, hidden_size) + query (B, T, hidden_size)
# Output: (B, T, hidden_size)
@registry._register_impl(_type='decoder', to_register=None, name='Decoder')
class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.MODEL.DECODER_LAYERS
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    # mask: BoolTensor with
    #   -  True for invalid locations, and
    #   -  False for valid locations
    def generate_mask(self, query):
        tgt_mask = torch.eye(query.shape[1]).type_as(query)
        tgt_mask = (1 - tgt_mask).bool()
        return tgt_mask

    def forward(self, query, memory, **kwargs):
        tgt_mask = self.generate_mask(query)
        output = self.transformer_decoder(query, memory, tgt_mask=tgt_mask, **kwargs) # (B, T, hidden_size)
        return output

# ------------------------------------------------------------------------------------------ #

@registry._register_impl(_type='head', to_register=None, name='DirClsHead')
class DirClsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.num_classes = cfg.MODEL.NUM_CLASSES 
        self.head = nn.Linear(hidden_dim, self.num_classes * 4) 

    def forward(self, query, dec_out, batch): # (B, T, D)
        pred = self.head(dec_out) # (B, T, O * 4)
        pred = rearrange(pred, 'b t (d o) -> b t d o', d=4, o=self.num_classes)
        return pred

@registry._register_impl(_type='head', to_register=None, name='DirDistClfHead')
class DirDistClfHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.num_classes = cfg.MODEL.NUM_DIST_BUCKETS * cfg.MODEL.NUM_CLASSES 
        self.head = nn.Linear(hidden_dim, self.num_classes * 4) 

    def forward(self, query, dec_out, batch): # (B, T, D)
        pred = self.head(dec_out) # (B, T, 4, O*bucket)
        pred = rearrange(pred, 'b t (d o) -> b t d o', d=4, o=self.num_classes)
        return pred