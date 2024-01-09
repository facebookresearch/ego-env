#!/usr/bin/env python3

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
from ..tasks.nav.exp_nav import ExplorationEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="Exploration-v1")
class ExplorationDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Exploration dataset.
    """

    episodes: List[ExplorationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)


    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # keep episodes for a single chunk
        valid_episodes = set(config.WALKTHROUGH_EPISODES)
        self.episodes = list(
            filter(
                lambda ep: (os.path.basename(ep.scene_id).split('.')[0], ep.episode_id) in valid_episodes,
                self.episodes
            )
        )

    def from_json(self, json_str, scenes_dir):
        deserialized = json.loads(json_str)
        for episode in deserialized["episodes"]:
            episode = ExplorationEpisode(**episode)
            episode.goals = [NavigationGoal(**goal) for goal in episode.goals]

            if not episode.scene_id.startswith(scenes_dir):
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            self.episodes.append(episode)
