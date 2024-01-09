#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
from habitat.config import Config as CN

CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.CMD_TRAILING_OPTS = []
_C.CHECKPOINT_DIR = 'cv/tmp'
_C.GPUS = 8

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.WALKTHROUGH_LENGTH = 512
_C.DATA.WINDOW_SIZE = 256
_C.DATA.MEMORY_SIZE = 64 
_C.DATA.OBS_MASK_PROB = 0.0
_C.DATA.TASK = '???'
_C.DATA.DATA_DIR = "data/walkthrough_data/hm3d/v1/"
_C.DATA.FEAT_SRC = "r50_feats"
_C.DATA.SRC = "hm3d"
_C.DATA.QUERIES_PER_BATCH = 1
_C.DATA.MULTITASK_WT = 0.1

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'EnvStatePredictor'
_C.MODEL.POS_ENCODER = 'SinCos'
_C.MODEL.POSE_DIM = 128
_C.MODEL.POSE_FREEZE = True
_C.MODEL.FRAME_ENCODER = 'FeatEncoder'
_C.MODEL.ENV_ENCODER = 'EnvEncoder'
_C.MODEL.DECODER = 'Decoder' 
_C.MODEL.HEAD = ['DirClsHead', 'DirDistClfHead']
_C.MODEL.IMG_FEAT_DIM = 2048 # R50
_C.MODEL.NUM_CLASSES = 1
_C.MODEL.NUM_DIST_BUCKETS = 5
_C.MODEL.HIDDEN_DIM = 128
_C.MODEL.ENCODER_LAYERS = 2
_C.MODEL.DECODER_LAYERS = 2
_C.MODEL.WEIGHTS = ''
_C.MODEL.POSE_MODEL_WEIGHTS = ''

_C.MODEL.POSE_EMBED = CN()
_C.MODEL.POSE_EMBED.FRAME_ENCODER = 'FeatEncoder'
_C.MODEL.POSE_EMBED.POS_ENCODER = 'SinCos'
_C.MODEL.POSE_EMBED.IMG_FEAT_DIM = 2048
_C.MODEL.POSE_EMBED.HIDDEN_DIM = 128
_C.MODEL.POSE_EMBED.NUM_LAYERS = 2

# -----------------------------------------------------------------------------
# Downstream options
# -----------------------------------------------------------------------------
_C.DOWNSTREAM = CN()
_C.DOWNSTREAM.VIS_FPS = 7.5 # 7.5 R50 frame feature per second (~4 stride @ 30FPS)
_C.DOWNSTREAM.ENV_FPS = 1 # 1 EgoEnv feature per second


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.BATCH_SIZE = 512
_C.OPTIM.LR = 1e-4
_C.OPTIM.WEIGHT_DECAY = 2e-5
_C.OPTIM.WARMUP_EPOCHS = 50
_C.OPTIM.MAX_EPOCHS = 2500
_C.OPTIM.MILESTONES = [10000]
_C.OPTIM.WORKERS = 8
_C.OPTIM.EVAL_VAL_EVERY = 10

# -----------------------------------------------------------------------------

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
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config