#!/usr/bin/env python3

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import attr
import numpy as np
import json
from gym import spaces
import math

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.errors import GreedyFollowerError
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
    quaternion_to_list
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat.tasks.nav.nav import (
    NavigationGoal,
    NavigationTask,
    MoveForwardAction,
)

# ------------------------------------------------------------------------------
# Sensors
# ------------------------------------------------------------------------------

@registry.register_sensor
class WorldPoseSensor(Sensor):
    r"""Sensor to return world position and rotation of the agent.
    Args:
        sim: reference to the simulator for calculating task observations.
    """
    cls_uuid: str = "world_pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (7,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode: Episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position.tolist()
        agent_rotation = quaternion_to_list(agent_state.rotation)

        return np.array(agent_position + agent_rotation)

@registry.register_sensor(name="PoseSensor")
class PoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame defined by the
    episode, i.e. the axis it faces along and the origin is defined by its state at
    t=0. Additionally contains the time-step of the episode.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_time = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_time = 0.0
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        ep_time = self._episode_time
        self._episode_time += 1.0

        return np.array(
            [-agent_position_xyz[2], agent_position_xyz[0], agent_heading[0], ep_time],
            dtype=np.float32
        )

@registry.register_sensor(name="CollisionSensor")
class CollisionSensor(Sensor):
    r"""Retruns 1 if a collision occurred in the previous action, otherwise it
    returns 0.
    Args:
        sim: reference to the simulator for calculating task observations.
    """
    cls_uuid: str = "collision_sensor"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32, )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self._current_episode_id != episode_uniq_id:
            self._current_episode_id = episode_uniq_id
            return np.array([0.0])
        if self._sim.previous_step_collided:
            return np.array([1.0])
        else:
            return np.array([0.0])

# ------------------------------------------------------------------------------
# Measures
# ------------------------------------------------------------------------------

@registry.register_measure
class CellsCovered(Measure):
    r"""Cells covered
    The measure evaluates the number of discrete (x, y) cells visited during
    an episode.
    """
    cls_uuid: str = "cells_covered"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config
        # Measure specific variables
        self._environment_cells = None
        self.L_min = None
        self.L_max = None
        self._metric = 0.0
        self.grid_size = config.GRID_SIZE
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = 0.0

        self.L_min, self.L_max = self._sim.pathfinder.get_bounds()
        map_size = (
            abs(self.L_max[0] - self.L_min[0]) / self.grid_size,
            abs(self.L_max[2] - self.L_min[2]) / self.grid_size,
        )
        map_size = (int(map_size[0])+1, int(map_size[1])+1)
        self._environment_cells = np.zeros(map_size)
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _convert_to_grid(self, position):
        """position - (x, y, z) in real-world coordinates """
        grid_x = (position[0] - self.L_min[0]) / self.grid_size
        grid_y = (position[2] - self.L_min[2]) / self.grid_size
        grid_x = int(grid_x)
        grid_y = int(grid_y)
        return (grid_x, grid_y)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        agent_position = self._sim.get_agent_state().position
        grid_x, grid_y = self._convert_to_grid(agent_position)
        self._environment_cells[grid_x, grid_y] = 1.0
        self._metric = self._environment_cells.sum()


@registry.register_measure
class ForwardSuccess(Measure):
    r"""Forward Success
    The measure evaluates the number of successful forward actions
    (i.e., forward actions with no collisions) throughout an episode.
    """
    cls_uuid: str = "forward_success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config
        self._metric = 0.0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = 0.0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if task.successful_forward_step:
            self._metric += 1.0


# ------------------------------------------------------------------------------
# Task
# ------------------------------------------------------------------------------

@attr.s(auto_attribs=True, kw_only=True)
class ExplorationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.
    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = []
    start_room: Optional[str] = None
    shortest_paths: Optional[List[ShortestPathPoint]] = None


@registry.register_task(name="Exp-v0")
class ExplorationTask(NavigationTask):
    r"""An Exploration Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.successful_forward_step = False

    def reset(self, episode: Type[Episode]):
        observations = super().reset(episode)
        self.successful_forward_step = False
        return observations

    def step(self, action: Union[int, Dict[str, Any]], episode: Type[Episode]):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]
        observations = task_action.step(**action["action_args"], task=self)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        # Update if forward action succeeded or not
        if action_name == MoveForwardAction.name and (not self._sim.previous_step_collided):
            self.successful_forward_step = True
        else:
            self.successful_forward_step = False

        return observations


import torch
import copy
from habitat_sim.nav import GreedyGeodesicFollower

@registry.register_task(name="OracleExp-v0")
class OracleExplorationTask(NavigationTask):
    r"""An Exploration Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.successful_forward_step = False

    def reset(self, episode: Type[Episode]):
        observations = super().reset(episode)
        self.successful_forward_step = False

        # compute the entire trajectory here
        self.goals = copy.deepcopy(episode.goals)
        self.follower = GreedyGeodesicFollower(
            self._sim.pathfinder,
            self._sim.get_agent(agent_id=self._sim._default_agent_id),
            self.goals[0].radius
        )
        self.goal_idx = 0

        return observations

    def step(self, action: Union[int, Dict[str, Any]], episode: Type[Episode]):

        next_action = None
        num_skips = 0
        while next_action is None:

            # force a move forward action
            if num_skips > 20:
                next_action = 1
                break

            # if all goals have been reached, shuffle goals and repeat
            if self.goal_idx == len(self.goals):
                np.random.shuffle(self.goals)
                self.goal_idx = 0

            goal = self.goals[self.goal_idx]

            try:
                next_action = self.follower.next_action_along(goal.position)
            except GreedyFollowerError:
                print (f'Greedy follower error. Skipping goal ({num_skips})')
                next_action = None
                num_skips += 1

            if next_action is None:
                self.goal_idx += 1

        # replace action with pre-computed path
        action = {'action': next_action - 1} # 1 = forward

        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]
        observations = task_action.step(**action["action_args"], task=self)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        # Update if forward action succeeded or not
        if action_name == MoveForwardAction.name and (not self._sim.previous_step_collided):
            self.successful_forward_step = True
        else:
            self.successful_forward_step = False

        return observations
