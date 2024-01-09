#!/usr/bin/env python3

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union
import collections
import numpy as np

from habitat_baselines.config.default import get_config as get_baseline_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","


_C = CN()

# -----------------------------------------------------------------------------
# SIMULATOR CONFIG
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TURN_ANGLE = 30  # angle to rotate left or right in degrees
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C.THREADED = False
_C.TASK_CONFIG = CN()
_C.TASK_CONFIG.TASK = CN()
_C.TASK_CONFIG.DATASET = CN()
_C.TASK_CONFIG.DATASET.WALKTHROUGH_SCENE = None
_C.TASK_CONFIG.DATASET.WALKTHROUGH_EPISODES = None 
# -----------------------------------------------------------------------------
# CELLS COVERED MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK_CONFIG.TASK.CELLS_COVERED = CN()
_C.TASK_CONFIG.TASK.CELLS_COVERED.TYPE = "CellsCovered"
_C.TASK_CONFIG.TASK.CELLS_COVERED.GRID_SIZE = 0.5
# -----------------------------------------------------------------------------
# FORWARD SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK_CONFIG.TASK.FORWARD_SUCCESS = CN()
_C.TASK_CONFIG.TASK.FORWARD_SUCCESS.TYPE = "ForwardSuccess"
# -----------------------------------------------------------------------------
# POSE SENSOR
# -----------------------------------------------------------------------------
_C.TASK_CONFIG.TASK.POSE_SENSOR = CN()
_C.TASK_CONFIG.TASK.POSE_SENSOR.TYPE = "PoseSensor"
# -----------------------------------------------------------------------------
# WORLD POSE SENSOR
# -----------------------------------------------------------------------------
_C.TASK_CONFIG.TASK.WORLD_POSE_SENSOR = CN()
_C.TASK_CONFIG.TASK.WORLD_POSE_SENSOR.TYPE = "WorldPoseSensor"
# -----------------------------------------------------------------------------
# COLLISION SENSOR 
# -----------------------------------------------------------------------------
_C.TASK_CONFIG.TASK.COLLISION_SENSOR = CN()
_C.TASK_CONFIG.TASK.COLLISION_SENSOR.TYPE = "CollisionSensor"
# -----------------------------------------------------------------------------
# OBJSTATE SENSOR 
# -----------------------------------------------------------------------------
_C.TASK_CONFIG.TASK.OBJSTATE_SENSOR = CN()
_C.TASK_CONFIG.TASK.OBJSTATE_SENSOR.TYPE = "ObjStateSensor"


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            returned = CN(returned)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = get_baseline_config(config_paths)
    config = deep_update(config, _C)

    if opts:
        config.defrost()
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)
        config.freeze()

    return config
