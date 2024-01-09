#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from habitat.core.registry import registry

from data.constants import OBJ_VOCAB, SCENE_SPLITS
from .utils import  _compute_relative_pose, strided_window_frames

class LocalStateDataset:

    def __init__(self, cfg):

        # ---- config entries ---- #
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_DIR
        self.src = cfg.DATA.SRC
        self.walkthrough_len = cfg.DATA.WALKTHROUGH_LENGTH
        self.mem_size = cfg.DATA.MEMORY_SIZE
        self.win_size = cfg.DATA.WINDOW_SIZE
        self.queries_per_batch = cfg.DATA.QUERIES_PER_BATCH
        self.vocab_map = {label: idx for idx, label in enumerate(OBJ_VOCAB[self.src])}
        # ------------------------ #

        episodes = self.load_episodes()  

        train_scenes = set(SCENE_SPLITS[self.src]['train'])
        val_scenes = set(SCENE_SPLITS[self.src]['val'])
        print (f'Train scenes: {len(train_scenes)} | Val scenes: {len(val_scenes)}')

        self.train_data = [entry for entry in episodes if entry['scene_id'] in train_scenes]
        self.val_data = [entry for entry in episodes if entry['scene_id'] in val_scenes]

        # pre-compute info for val entries that are normally randomly sampled for train (e.g., query_inds)
        # also generate multiple val instances from a single instance 
        self.val_data = self.precompute_val_queries(self.val_data)

        print (f'Train instances: {len(self.train_data)} | Val instances: {len(self.val_data)}')

    def load_episodes(self):
        episodes = torch.load(f'{self.data_dir}/episode_list.pth')  
        return episodes

    # select K query_inds per val walkthrough uniformly 
    # val instances are always single query
    def precompute_val_queries(self, val_data, K=4):
        val_data_with_queries = []   
        for entry in val_data:
            query_inds = torch.linspace(0, self.walkthrough_len-1, K).long().tolist()
            for idx in query_inds:
                val_data_with_queries.append({
                    **entry,
                    'query_inds': np.array([idx])
                })
        print (f'Resampled val_data: {len(val_data)} --> {len(val_data_with_queries)}')
        return val_data_with_queries

    def set_mode(self, mode):
        self.mode = mode
        self.data = {'train': self.train_data, 'val': self.val_data}[self.mode]
        print ('Setting mode', self.mode)
        return self

    def load_state_data(self, entry, state_type):
        scene_id, episode_id = entry['scene_id'], entry['episode_id']
        state_fl = f'{self.data_dir}/state/{state_type}/{scene_id}_{episode_id}.pth'
        feats = torch.load(state_fl) # (T, ...)
        return feats

    # Returns:
    #   - A label tensor for each query_idx
    #   - A dict with any other label related metadata
    def get_state_label(self, query_inds, frame_inds, entry):
        raise NotImplementedError

    def get_frame_inds(self, entry, query_inds):
        frame_inds, mem_blocks, rel_frame_pos, rel_query_pos = strided_window_frames(
            query_inds, self.walkthrough_len, self.mem_size, self.win_size, self.mode=='train'
        )
        return frame_inds, mem_blocks, rel_frame_pos, rel_query_pos

    def get_query_inds(self, entry):
        query_inds = entry.get('query_inds', np.arange(0, self.walkthrough_len)) # if val, already pre-selected
        if self.mode=='train':
            replace = len(query_inds) < self.queries_per_batch
            query_inds = np.random.choice(query_inds, self.queries_per_batch, replace=replace)
        query_inds = torch.from_numpy(query_inds).long()
        return query_inds


    def __getitem__(self, index):
        entry = self.data[index]
        scene_id, episode_id = entry['scene_id'], entry['episode_id']

        query_inds = self.get_query_inds(entry)
        frame_inds, mem_blocks, rel_frame_pos, rel_query_pos = self.get_frame_inds(entry, query_inds)

        traj_frames = self.load_state_data(entry, self.cfg.DATA.FEAT_SRC)
        frames = traj_frames[frame_inds]
        query = traj_frames[query_inds]

        labels = self.get_state_label(query_inds, frame_inds, entry) 

        instance = {
            'idx': index,
            'rgb': frames, 
            'query_rgb': query,
            'mem_blocks': mem_blocks,
            'frame_inds': (rel_frame_pos * (self.walkthrough_len - 1)).long(),
            'query_inds': (rel_query_pos * (self.walkthrough_len - 1)).long(),
            **labels
        }

        return instance

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        keys = batch[0].keys()
        transposed_batch = {key: [item[key] for item in batch] for key in keys}
        for key in transposed_batch:
            if key in ['rgb']:
                transposed_batch[key] = pad_sequence(transposed_batch[key], batch_first=True)
            else:
                transposed_batch[key] = default_collate(transposed_batch[key])
        return transposed_batch


@registry._register_impl(_type='dataset', to_register=None, name='PoseEmbedding')
class PoseDataset(LocalStateDataset):

    # Relative pose is calculate for all pairs
    def precompute_val_queries(self, val_data, K=4):
        return val_data

    def get_pose_data(self, entry, frame_inds):

        if 'world_pose' not in entry:
            entry['world_pose'] = self.load_state_data(entry, 'pose')

        traj_pose = entry['world_pose'][frame_inds] # (S, 3)
        S, T = traj_pose.shape[0], traj_pose.shape[0]

        rel_pose = _compute_relative_pose(
            traj_pose.repeat(1, S).view(S*T, -1), # [qpose0, qpose0 ... , qpose1, qpose1, ...]
            traj_pose.repeat(T, 1), # [tpose0, tpose1, ..., tpose0, tpose1, ...]
        ) # (S*T, 3)
        rel_pose = rel_pose.view(T, S, -1) # (T, S, 3)

        rel_xz = rel_pose[:, :, :2] / 5
        rel_theta = rel_pose[:, :, 2] # (T, S)
        rel_R = rel_pose[:, :, :2].norm(dim=2) # (T, S)

        # discretize
        R_buckets = torch.tensor([0.5, 2.0, 4.0, 8.0]) 
        drel_R = torch.bucketize(rel_R, R_buckets).long() # 4 + 1

        drel_theta = rel_theta + np.pi # [0, 2pi]
        drel_theta = drel_theta.clip(0, 2*np.pi)
        drel_theta = np.rad2deg(drel_theta) % 360
        drel_theta = (drel_theta/45).long() # 8

        theta_invalid = rel_R > 8.0
        drel_theta[theta_invalid] = -1

        return {
            'rel_R_disc': drel_R,
            'rel_theta_disc': drel_theta
        }

    def __getitem__(self, index):
        entry = self.data[index]
        scene_id, episode_id = entry['scene_id'], entry['episode_id']

        query_inds = torch.randint(0, self.walkthrough_len, (1,))
        frame_inds, mem_blocks, rel_frame_pos, rel_query_pos = self.get_frame_inds(entry, query_inds)
        frame_inds = frame_inds[mem_blocks[0]] # (S,)
        rel_frame_pos = rel_frame_pos[0] # (S,)

        frames = self.load_state_data(entry, self.cfg.DATA.FEAT_SRC)[frame_inds]

        instance = {
            'idx': index,
            'rgb': frames,
            'frame_inds': (rel_frame_pos * (self.walkthrough_len - 1)).long(),
            **self.get_pose_data(entry, frame_inds)
        }

        # mask for transformer and labels
        mask = torch.zeros(len(frames)).bool()
        if self.mode == 'train':
            mask = torch.rand(len(frames)) < self.cfg.DATA.OBS_MASK_PROB # 20% true = ignore
            instance['rel_R_disc'][mask, :] = -1
            instance['rel_R_disc'][:, mask] = -1
            instance['rel_theta_disc'][mask, :] = -1
            instance['rel_theta_disc'][:, mask] = -1

        instance['obs_mask'] = mask

        return instance

#---------------------------------------------------------------------------------------------------#
# Labels = obj_labels (T, 4, #obj) + dist_labels (T, 4, #obj)
# Binary tensor for each of the 4 cardinal directions indicating objects visible.
# Categorical tensor for each of the 4 cardinal objects indicating discretized distance of each object.
# T queries per walkthrough clip
#---------------------------------------------------------------------------------------------------#

@registry._register_impl(_type='dataset', to_register=None, name='CardinalObjectState')
class CardinalObjectState(LocalStateDataset):

    def load_episodes(self):
        episodes = torch.load(f'{self.data_dir}/episode_list.pth')  
        episodes = [ep for ep in episodes if os.path.exists(f"{self.data_dir}/state/cardinal_object_state/{ep['scene_id']}_{ep['episode_id']}.pth")]
        return episodes

    def get_state_label(self, query_inds, frame_inds, entry):
        scene_id, episode_id = entry['scene_id'], entry['episode_id']
        state_fl = f'{self.data_dir}/state/cardinal_object_state/{scene_id}_{episode_id}.pth'
        state_labels, state_dists = torch.load(state_fl) # (W, 4, #objs) x2
        query_label = state_labels[query_inds] # (T, 4, #objs)
        query_dists = state_dists[query_inds]
        return {'labels': query_label, 'dists': query_dists}

