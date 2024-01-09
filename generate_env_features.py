#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import argparse
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchvision.models as tmodels
import torch.nn as nn
from PIL import Image
import tqdm

import decord
decord.bridge.set_bridge('torch')

from state_prediction.config.defaults import get_config
from state_prediction.data.utils import strided_window_frames
from state_prediction.models.env_model import EnvStatePredictor


class FrameDataset:
    def __init__(self, video, cfg):

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: Image.fromarray(x)),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        vr = decord.VideoReader(video)
        fps = vr.get_avg_fps()
        stride = int(fps / cfg.DOWNSTREAM.VIS_FPS)

        frame_inds = np.arange(0, len(vr), stride)
        self.frames = vr.get_batch(frame_inds).numpy()
        print (f'Generating frame features for {video}. N = {len(self.frames)} (fps: {fps}, stride: {stride})')

    def __getitem__(self, index):
        frame = self.frames[index] # (H, W, 3)
        frame = self.transform(frame)
        return frame

    def __len__(self):
        return len(self.frames)



class EnvDataset:

    def __init__(self, video, frame_feats, cfg):
        self.cfg = cfg
        self.win_size = cfg.DATA.WINDOW_SIZE
        self.mem_size = cfg.DATA.MEMORY_SIZE
        self.walkthrough_len = cfg.DATA.WALKTHROUGH_LENGTH

        vr = decord.VideoReader(video)
        fps = vr.get_avg_fps()
        N_env_feats = int(len(vr) / fps * cfg.DOWNSTREAM.ENV_FPS)
        N_frame_feats = frame_feats.shape[0]

        query_inds = np.linspace(0, 1, N_env_feats) * (N_frame_feats - 1)
        self.query_inds = query_inds.astype(int).tolist()
       
        self.frame_feats = frame_feats
        self.T = self.frame_feats.shape[0]

    def __getitem__(self, index):
        query_idx = self.query_inds[index]
        query_inds = torch.LongTensor([query_idx])

        frame_inds, mem_blocks, rel_frame_pos, rel_query_pos = strided_window_frames(
            query_inds, self.T, self.mem_size, self.win_size, False
        )                 

        instance = {
            'rgb': self.frame_feats[frame_inds], # (S, 2048)
            'query_rgb': self.frame_feats[query_idx].unsqueeze(0), # (1, 2048)
            'mem_blocks': mem_blocks,
            'frame_inds': (rel_frame_pos * (self.walkthrough_len - 1)).long(),
            'query_inds': (rel_query_pos * (self.walkthrough_len - 1)).long(),
        }
        return instance

    def __len__(self):
        return len(self.query_inds)

    def collate_fn(self, batch):
        keys = batch[0].keys()
        transposed_batch = {key: [item[key] for item in batch] for key in keys}
        for key in transposed_batch:
            if key in ['env_rgb']:
                transposed_batch[key] =  pad_sequence(transposed_batch[key], batch_first=True)
            else:
                transposed_batch[key] = default_collate(transposed_batch[key])
        return transposed_batch

def generate_frame_feats(args, cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = tmodels.resnet50(pretrained=True)
    net.fc = nn.Identity()
    net.eval().to(device)

    dataset = FrameDataset(args.video, cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    features = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc='frames'):
        batch = batch.to(device)
        with torch.no_grad():
            feat = net(batch).cpu() # (B, 2048)
        features.append(feat)
    features = torch.cat(features, 0) # (N, 2048)
    
    return features

def generate_env_feats(frame_feats, args, cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = EnvStatePredictor(cfg)
    net.eval().to(device)

    dataset = EnvDataset(args.video, frame_feats, cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)

    features = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc='env clips'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            feat, _ = net.env_forward(batch)
        features.append(feat.cpu())
    features = torch.cat(features, 0)[:, 0] # (N, D)

    print ('Generated env features:', features.shape)
    if args.save:
        torch.save(features, args.save)

    return features


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='config yaml for experiment')
    parser.add_argument('--video', default='video.mp4', help='Video to generate env features for')
    parser.add_argument('--save', default=None, help='path to save environment features to (optional)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")
    args = parser.parse_args()

    cfg = get_config(args.config, args.opts)
    frame_features = generate_frame_feats(args, cfg)
    env_features = generate_env_feats(frame_features, args, cfg)



