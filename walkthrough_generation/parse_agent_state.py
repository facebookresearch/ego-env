#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import glob
import os
import re
import torch
from joblib import Parallel, delayed
import tqdm
import torch
import numpy as np
import argparse
import collections

from habitat.core.registry import registry

from data.constants import OBJ_VOCAB
from util import pose_util

@registry._register_impl(_type='summarize', to_register=None, name='episodes')
def summarize_episodes(args):
    scene_pat = re.compile('(.*)_(.*).pth')
    files = list(glob.glob(f'{args.data_dir}/state/info/*.pth'))

    episodes_by_scene = collections.defaultdict(list)
    for fl in tqdm.tqdm(files):
        name = os.path.basename(fl)
        scene_id, episode_id = scene_pat.match(name).groups()
        episodes_by_scene[scene_id].append({'scene_id': scene_id, 'episode_id':episode_id})

    # sample K episodes per environment
    rs = np.random.RandomState(args.seed)
    K = 32
    episodes = []
    for scene in episodes_by_scene:
        rs.shuffle(episodes_by_scene[scene])
        episodes += episodes_by_scene[scene][:K]

    print (f'{len(episodes)} episodes')
    torch.save(episodes, f'{args.data_dir}/episode_list.pth')


@registry._register_impl(_type='summarize', to_register=None, name='pose')
def summarize_pose(args):

    def generate(episode):
        scene_id, episode_id = episode['scene_id'], episode['episode_id']
        info = torch.load(f'{args.data_dir}/state/info/{scene_id}_{episode_id}.pth')
        pose_data = torch.Tensor(list(map(pose_util.parse_pose, info['world_poses'])))
        torch.save(pose_data, f'{args.data_dir}/state/pose/{scene_id}_{episode_id}.pth')

    os.makedirs(f'{args.data_dir}/state/pose/', exist_ok=True)
    episodes = torch.load(f'{args.data_dir}/episode_list.pth')
    Parallel(n_jobs=64, verbose=1)(delayed(generate)(ep) for ep in episodes)


@registry._register_impl(_type='summarize', to_register=None, name='cardinal_object_state')
def summarize_cardinal_object_state(args):

    def generate(episode):
        scene_id, episode_id = episode['scene_id'], episode['episode_id']

        df = torch.load(f'{args.data_dir}/state/detected_objects/{scene_id}_{episode_id}.pth')
        if len(df) == 0:
            return

        # detected_objects
        if 'quadrant' not in df:
            df['quadrant'] = df[[f'v_{dir_idx}' for dir_idx in range(4)]].to_numpy().argmax(1)
            df['dist'] = df[[f'dist_{dir_idx}' for dir_idx in range(4)]].to_numpy().max(1)

        df = df[df['category'].isin(vocab_map)]
        df['cat_id'] = df['category'].apply(lambda obj_cat: vocab_map[obj_cat])

        objs = torch.zeros(512, 4, len(vocab_map))
        dist = torch.zeros(512, 4, len(vocab_map)) - 1
        for entry in df.to_dict('records'):
            for dir_idx in range(4):
                objs[entry['t'], dir_idx, entry['cat_id']] = max(objs[entry['t'], dir_idx, entry['cat_id']], entry[f'v_{dir_idx}'])
            dist[entry['t'], entry['quadrant'], entry['cat_id']] = entry['dist']

        mask = (objs > 0) & (dist != -1) & (dist < 5)
        obj_labels = mask.float()
        if torch.all(obj_labels==0):
            return

        dist_buckets = torch.tensor([0.25, 1.0, 2.0, 4.0]) 
        dist_labels = torch.bucketize(dist, dist_buckets).long() # 4 + 1
        dist_labels[~mask] = -1 

        torch.save([obj_labels, dist_labels], f'{args.data_dir}/state/cardinal_object_state/{scene_id}_{episode_id}.pth')

    os.makedirs(f'{args.data_dir}/state/cardinal_object_state/', exist_ok=True)
    vocab_map = {label: idx for idx, label in enumerate(OBJ_VOCAB['hm3d'])}
    episodes = torch.load(f'{args.data_dir}/episode_list.pth')
    Parallel(n_jobs=64, verbose=1)(delayed(generate)(ep) for ep in episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default='data/walkthrough_data/hm3d/v1/')
    parser.add_argument("--mode", type=str, default=None, help='episodes|pose|cardinal_object_state')
    args = parser.parse_args()

    registry._get_impl('summarize', args.mode)(args)
        



