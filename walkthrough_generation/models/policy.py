#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.config import Config
from gym import spaces
import torch
import random

from ..tasks.nav.exp_nav import (
    CollisionSensor,
)

class PointNavRoombaNet(Net):
    r"""Network that keeps moving forward till it collides. Upon collision, it turns
    left or right by a random amount.
    """

    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()
        self.action_space = action_space
        self.train()
        self.max_rotations = 5
        self.num_rotations = None
        self.rotations_count = None
        self.rotation_action = None
        self.random_stop_prob = 0.1

        if action_space.n == 4:
            self.STOP_ACTION = 0
            self.MOVE_FORWARD = 1
            self.TURN_LEFT = 2
            self.TURN_RIGHT = 3
        else:
            self.STOP_ACTION = None
            self.MOVE_FORWARD = 0
            self.TURN_LEFT = 1
            self.TURN_RIGHT = 2

    @property
    def memory_dim(self):
        return 1

    @property
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        assert CollisionSensor.cls_uuid in observations
        collisions = observations[CollisionSensor.cls_uuid]
        assert collisions.shape[0] == 1, "Batch size must be 1"

        if self.rotations_count is None:
            if collisions[0, 0].item() == 1:
                # If collision occurred, pick a rotation action to execute
                self.num_rotations = random.randint(1, self.max_rotations)
                self.rotations_count = 0
                self.rotation_action = random.choice([self.TURN_LEFT, self.TURN_RIGHT])
            else:
                # If not collision, move forward
                return torch.Tensor([[self.MOVE_FORWARD]]).to(collisions.device).long()

        self.rotations_count += 1
        actions = torch.Tensor([[self.rotation_action]]).to(collisions.device).long()
        if self.rotations_count == self.num_rotations:
            self.rotations_count = None
            self.rotation_action = None
            self.num_rotations = None
        
        if self.STOP_ACTION is not None:
            if random.random() <= self.random_stop_prob:
                actions[:] = self.STOP_ACTION

        return actions


@baseline_registry.register_policy(name="PointNavRoombaPolicy")
class PointNavRoombaPolicy(NetPolicy):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(
            PointNavRoombaNet(
                observation_space,
                action_space,
                **kwargs
            ),
            action_space.n,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        actions = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        return None, actions, None, rnn_hidden_states


    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
        )