#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import tqdm
import argparse
import numpy as np
from PIL import Image
import pandas as pd
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import torchvision.io as tio
from torchvision import transforms
import habitat_sim
from habitat.core.registry import registry

from mmdet.apis import init_detector, inference_detector
import mmcv

from util import pose_util

def make_habitat_configuration(scene_path, sensors, scene_dataset_config):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    if scene_dataset_config is not None:
        backend_cfg.scene_dataset_config_file = scene_dataset_config

    sensor_types = {
        'rgb': habitat_sim.SensorType.COLOR,
        'semantic': habitat_sim.SensorType.SEMANTIC,
        'depth': habitat_sim.SensorType.DEPTH
    }
    
    sensor_specifications = []
    for sensor in sensors:
        sensor_cfg = habitat_sim.bindings.CameraSensorSpec()
        sensor_cfg.uuid = sensor
        sensor_cfg.resolution = [300, 300]
        sensor_cfg.hfov = 90
        sensor_cfg.sensor_type = sensor_types[sensor]
        sensor_specifications.append(sensor_cfg)
   
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specifications
    agent_cfg.action_space = {
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=90.0)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=90.0)
        )
    }

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

def load_sim(scene_path, sensors, scene_dataset_config):
    sim_cfg = make_habitat_configuration(scene_path, sensors, scene_dataset_config)
    hsim = habitat_sim.Simulator(sim_cfg)
    return hsim

class BaseExtractor:
    def __init__(self, args):
        self.args = args

    def collate(self, states):
        raise NotImplementedError

    def save(self, data, out_dir, scene_id, episode_id):
        torch.save(data, f'{out_dir}/{scene_id}_{episode_id:04d}.pth')


@registry._register_impl(_type='state', to_register=None, name='detected_objects')
class DetectedObjs(BaseExtractor):

    def __init__(self, args):
        super().__init__(args)
        config_path = 'walkthrough_generation/models/mmdet_qinst_hm3d_config.py'
        model_path = 'walkthrough_generation/models/mmdet_qinst_hm3d.pth'
        self.detector = init_detector(config_path, model_path, args.gpu)
        self.score_thr = 0.5
        self.classes = self.detector.CLASSES

    def __call__(self, observations):

        # rgb_list is a list of HxWx3 uint8 RGB numpy arrays (one frame per direction)
        bgr_list = [obs['rgb'][:, :, :-1] for obs in observations] # (H, W, 3)
        rgb_list = [x[:, :, ::-1] for x in bgr_list] # BGR --> RGB
        predictions = inference_detector(self.detector, rgb_list)

        depths = [obs['depth'] for obs in observations]
        depth_masks = [(depth > 0) & (depth < 10) for depth in depths]

        curr_states = []
        for dir_id, (bbox_result, segm_result) in enumerate(predictions):

            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            segms = mmcv.concat_list(segm_result)
            if segms is None:
                continue
            segms = np.stack(segms, axis=0) # (N, 300, 300)

            bboxes = np.vstack(bbox_result)
            assert bboxes.shape[1] == 5, 'Malformed detection results'
            
            scores = bboxes[:, -1]
            inds = scores > self.score_thr

            scores = scores[inds]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            segms = segms[inds, ...]

            if labels.shape[0] == 0:
                continue

            depth = depths[dir_id]
            depth_mask = depth_masks[dir_id]
            for det_id in range(len(labels)):

                det_mask = segms[det_id] & depth_mask
                if det_mask.sum() == 0:
                    continue
                det_depth = depth[det_mask].mean()

                label = self.classes[labels[det_id]]
                vis = segms[det_id].sum() / (300 * 300)
                score = scores[det_id]

                oinfo = {}
                oinfo.update({f'v_{d}': 0 for d in range(4)})
                oinfo.update({f'dist_{d}': -1 for d in range(4)})
                oinfo.update({
                    'category': label,
                    f'v_{dir_id}': vis,
                    'score': score,
                    f'dist_{dir_id}': det_depth,
                })
                curr_states.append(oinfo)
        
        return curr_states

    def collate(self, states):
        data = []
        for t, obj_state in enumerate(states):
            data += [{'t':t, **obj} for obj in obj_state]
        df = pd.DataFrame(data)
        return df


@registry._register_impl(_type='state', to_register=None, name='rgb')
class RGBVideo(BaseExtractor):

    def __init__(self, args):
        super().__init__(args)
        self.device = args.gpu
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: Image.fromarray(x)),
        ])

    def __call__(self, observations):
        rgb = observations[0]['rgb'][:, :, :-1] # remove alpha channel --> (H, W, 3)
        rgb = self.transform(rgb)
        return rgb

    def collate(self, states):
        return states

    def save(self, data, out_dir, scene_id, episode_id):
        data = torch.stack([transforms.ToTensor()(img) for img in data], 0)
        data = (data*255).long()
        data = rearrange(data, 't c h w -> t h w c')
        tio.write_video(
            f'{out_dir}/{scene_id}_{episode_id:04d}.mp4',
            data,
            fps=10,
        )
        
class ImageFeatures(BaseExtractor):

    def __init__(self, args, net):
        super().__init__(args)
        self.device = args.gpu
        self.net = net
        self.net.eval()
        self.net.to(self.device)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: Image.fromarray(x)),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, observations):
        obs = observations[0] # only forward observation
        rgb = obs['rgb'][:, :, :-1] # remove alpha channel --> (H, W, 3)
        rgb = self.transform(rgb)
        rgb = rgb.unsqueeze(0) # (1, 3, H, W)
        with torch.no_grad():
            feats = self.net(rgb.to(self.device)).cpu()[0] # (D,)
        return feats

    def collate(self, states):
        return torch.stack(states, 0) # (T, D)


@registry._register_impl(_type='state', to_register=None, name='r50_feats')
class ResNet50Feats(ImageFeatures):
    def __init__(self, args):
        net = tmodels.resnet50(pretrained=True)
        net.fc = nn.Identity()
        super().__init__(args, net)

class StateExtractor:

    def __init__(self, args):
        self.args = args
        self.extractors = {state_type: registry._get_impl('state', state_type)(args) for state_type in args.state_types}
        for state_type in self.extractors:
            os.makedirs(f'{self.args.data_dir}/state/{state_type}', exist_ok=True)

    def parse_observation(self, obs, agent):
        arot_q = agent.state.rotation
        heading = pose_util._quat_to_xy_heading(arot_q)
        heading = np.rad2deg(heading)
        pose = agent.state.sensor_states['rgb'].position.tolist() + [heading]
        return {'pose': pose, **obs}

    def geodesic_distance(self, sim, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(position_a, dtype=np.float32) # (3,)
        path.requested_end = np.array(position_b, dtype=np.float32) # (N, 3)
        sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def extract_agent_state(self, episode):

        scene_id = episode['scene_id'].split("/")[-1].split(".")[0]
        episode_id = episode['episode_id']

        # Load simulator and agent
        sim = load_sim(episode['scene_id'], args.sensors, args.scene_dataset_config)
        semantic_scene = sim.semantic_scene
        agent = sim.initialize_agent(0)

        # Get list of all objects in scene
        objects_info = []
        for obj in semantic_scene.objects:
            if obj is None:
                continue

            obj_cat = obj.category.name()
            obj_id = int(obj.id.split('_')[1])
            
            objects_info.append({
                'id': obj_id,
                'category': obj_cat,
                'position': obj.obb.center, 
                'rotation': obj.obb.rotation,
            })

        state_data = {state_type: [] for state_type in self.extractors}
        for t, agent_pose in tqdm.tqdm(enumerate(episode['world_poses']), total=len(episode['world_poses'])):

            # move to position
            apos, arot = agent_pose[:3], agent_pose[3:]
            agent_state = habitat_sim.AgentState(
                position=apos,
                rotation=arot,
            )
            agent.set_state(agent_state)

            # update the object_info with geodesic distances
            obj_info = copy.deepcopy(objects_info)
            for obj in obj_info:
                obj['gdist'] = self.geodesic_distance(
                    sim, apos, obj['position']
                )

            # get observations for 4 directions
            obs = sim.get_sensor_observations()
            observations = [self.parse_observation(obs, agent)]
            for _ in range(4):
                obs = sim.step("turn_right")
                observations.append(self.parse_observation(obs, agent))
            observations = observations[:-1] # last obs is the same as first

            for state_type in self.extractors:
                step_state = self.extractors[state_type](observations)
                state_data[state_type].append(step_state)

        sim.close()

        for state_type in state_data:
            out_dir = f'{self.args.data_dir}/state/{state_type}'
            data = self.extractors[state_type].collate(state_data[state_type])
            self.extractors[state_type].save(data, out_dir, scene_id, episode_id)

def get_incomplete(episodes, args):
    incomplete = []
    for episode in tqdm.tqdm(episodes):
        scene_id, episode_id = episode['scene_id'], episode['episode_id']
        exists = [f'{args.data_dir}/state/{state_type}/{scene_id}_{episode_id}.pth' for state_type in args.state_types]
        if not np.all(list(map(os.path.exists, exists))):
            incomplete.append(episode)
    print (f'{len(incomplete)}/{len(episodes)} episodes remaining.')
    return incomplete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--job_id", type=int, default=None)
    parser.add_argument("--scene_dataset_config", default=None)
    parser.add_argument("--sensors", default=['rgb', 'depth'])
    parser.add_argument("--state_types", default=['rgb', 'r50_feats', 'detected_objects'])
    args = parser.parse_args()

    
    episodes = torch.load(f'{args.data_dir}/episode_list.pth')
    rs = np.random.RandomState(10)
    rs.shuffle(episodes)

    print(f"Number of episodes in {args.data_dir}: {len(episodes)}")

    chunksize = 100
    chunks = [episodes[idx:idx+chunksize] for idx in range(0, len(episodes), chunksize)]
    print (f'# chunks = {len(chunks)}')

    extractor = StateExtractor(args)
    for episode in chunks[args.job_id]:
        scene_id, episode_id = episode['scene_id'], episode['episode_id']
        info = torch.load(f'{args.data_dir}/state/info/{scene_id}_{episode_id}.pth')
        extractor.extract_agent_state(info)
