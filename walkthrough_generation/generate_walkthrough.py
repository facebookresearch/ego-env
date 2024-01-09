#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import json
import argparse
import gzip
import torch
import tqdm
import collections

from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import batch_obs

from .config.default import get_config
from .datasets import exploration_dataset
from .models import policy


class WalkthroughGenerator(PPOTrainer):
    r"""
    Generates agent walkthrough data.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    # modified from def _eval_checkpoint(...)
    def generate_walkthrough(self):

        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")
        
        config = self.config.clone()
        ppo_cfg = config.RL.PPO        
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
        config.freeze()

        # Save full trajectory information
        os.makedirs(f'{self.config.VIDEO_DIR}/info/', exist_ok=True)

        self._init_envs(config, is_eval=True)
        self.policy_action_space = self.envs.action_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg)
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )

        world_pose_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        episodes_count = 0
        while (
            episodes_count < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

          
            next_episodes = self.envs.current_episodes()
            n_envs = self.envs.num_envs
            for i in range(n_envs):
    
                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    scene_id = current_episodes[i].scene_id
                    episode_id = current_episodes[i].episode_id
                    scene_key = scene_id.split("/")[-1].split(".")[0]
                    video_info_path = f"info/{scene_key}_{episode_id:04d}.pth"
                    video_info = {
                        "world_poses": [pose.tolist() for pose in world_pose_frames[i]],
                        "episode_id": episode_id,
                        "scene_id": scene_id,
                    }
                    torch.save(
                        video_info,
                        os.path.join(self.config.VIDEO_DIR, video_info_path)
                    )
                    world_pose_frames[i] = []
                    episodes_count += 1

                # episode continues
                else:
                    world_pose_frames[i].append(observations[i]["world_pose"])

        self.envs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkthrough_dir", required=True, help='path to walkthrough episode specifications')
    parser.add_argument("--job_id", type=int, default=-1, help='chunk indexed by {SLURM_ARRAY_TASK_ID}')
    parser.add_argument("--exp_config", required=True, help='path following agent config.yaml')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    # Load all episode specifications
    datasetfile_path = config.TASK_CONFIG.DATASET.DATA_PATH.format(split=config.EVAL.SPLIT)
    episodes = json.load(gzip.open(datasetfile_path, "rt"))['episodes']

    episodes_by_scene = collections.defaultdict(list)
    for episode in episodes:
        episode_key = (os.path.basename(episode['scene_id']).split('.')[0], episode['episode_id'])
        episodes_by_scene[episode['scene_id']].append(episode_key)

    chunksize = 100
    N_chunks = 0
    chunks_by_scene = collections.defaultdict(list)
    for scene_id in episodes_by_scene:
        chunks = [episodes_by_scene[scene_id][idx:idx+chunksize] for idx in range(0, len(episodes_by_scene[scene_id]), chunksize)]
        chunks_by_scene[scene_id] += chunks
        N_chunks += len(chunks)
    
    # interleave chunks by {scene_id} so that episodes are generated uniformly
    chunks = []
    while len(chunks) < N_chunks:
        for scene in list(chunks_by_scene.keys()):
            if len(chunks_by_scene[scene]) > 0:
                chunks.append(chunks_by_scene[scene].pop())    
    print ('# Episode chunks = ', N_chunks) 

    # Generate walkthrough for all episodes in a chunk
    if args.job_id >= 0:
    
        config.defrost()
        config.TASK_CONFIG.DATASET.WALKTHROUGH_EPISODES = chunks[args.job_id]
        config.freeze()

        walkthrough_generator = WalkthroughGenerator(config=config)
        walkthrough_generator.generate_walkthrough()