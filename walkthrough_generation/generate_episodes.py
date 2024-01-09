#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import cv2
import glob
import gzip
import json
import numpy as np
import copy
from joblib import Parallel, delayed
import os
import tqdm
from sklearn.cluster import KMeans
import torch
import torchvision.utils

import habitat
from habitat.utils.visualizations import maps


# Load scene data for HM3D scenes
# Output: {scene_key: scene_data}
def load_scene_episodes(root_dir, splits):

    scene_episodes = {}
    for split in splits:
        scenes = glob.glob(f'{root_dir}/{split}/content/*.json.gz')
        for scene_path in tqdm.tqdm(scenes):

            scene_key = os.path.basename(scene_path).split('.json.gz')[0]
            with gzip.open(scene_path, "rt") as fp:
                episodes = json.load(fp)["episodes"]
            assert scene_key == episodes[0]["scene_id"].split("/")[-1].split(".")[0], 'Scene key mismatch'

            scene_episodes[scene_key] = episodes

    print (f'Number of scenes: {len(scene_episodes)}')
    return scene_episodes

def assign_clusters_from_area_hm3d(area):
    if area < 10.0:
        num_clusters = 2
    elif area < 30.0:
        num_clusters = 4
    elif area < 100.0:
        num_clusters = 16
    else:
        num_clusters = 32
    return num_clusters


def visualize_episode_maps(floor_episodes, floor_map, sim, out_fl):

    os.makedirs(os.path.dirname(out_fl), exist_ok=True)
    np.random.shuffle(floor_episodes)

    s = 10 # marker size
    grid_size = 3 # NxN grid of episodes
    colors = {
        'grey': np.array([100, 100, 100]),
        'green': np.array([255, 0, 0]),
        'blue': np.array([0, 255, 0])
    }

    floor_map_rgb = np.stack([floor_map, floor_map, floor_map], 2)
    floor_map_rgb = (floor_map_rgb * 255.0).astype(np.uint8)

    # Cluster centers (grey)
    for episode in floor_episodes:
        x, _, z = episode["start_position"]
        tx, ty = maps.to_grid(z, x, floor_map.shape, sim)
        floor_map_rgb[tx-s:tx+s, ty-s:ty+s] = colors['grey']

    episode_maps = []
    for episode in floor_episodes[0:grid_size*grid_size]:

        # selected goals
        episode_map = floor_map_rgb.copy()
        for goal in episode['goals']:
            x, _, z = goal["position"]
            tx, ty = maps.to_grid(z, x, floor_map.shape, sim)
            episode_map[tx-s:tx+s, ty-s:ty+s] = colors['blue']

        # start position
        x, _, z = episode["start_position"]
        tx, ty = maps.to_grid(z, x, floor_map.shape, sim)
        episode_map[tx-s:tx+s, ty-s:ty+s] = colors['green']

        episode_map = cv2.resize(episode_map, (256, 256))
        episode_map = torch.from_numpy(episode_map[:, :, [2, 0, 1]]) # BGR --> RGB
        episode_map = episode_map.permute(2, 0, 1)/255 # (3, 256, 256)

        episode_maps.append(episode_map)

    grid = torchvision.utils.make_grid(episode_maps, nrow=grid_size)
    torchvision.utils.save_image(grid, out_fl)

def generate(episodes, out_dir, seed, debug=False):

    scene_id = episodes[0]["scene_id"]
    scene_key = scene_id.split("/")[-1].split(".")[0]
    scene_path = f"./data/scene_datasets/{scene_id}"

    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene_path
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.SIMULATOR.AGENT_0.HEIGHT = 1.5
    cfg.SIMULATOR.AGENT_0.RADIUS = 0.1
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    num_goals = [8] # how many points to navigate to in sequence
    num_resample = 2 # how many paths to sample for start point
    map_resolution = 1250 # floor map resolution
    min_episode_threshold = 4 # skip if there are too few episodes
    rs = np.random.RandomState(seed)

    floor_episode_data = []
    for episode in episodes:
        start_height = episode["start_position"][1]
        found_match = False
        for floor_idx, episode_data in enumerate(floor_episode_data):
            if abs(episode_data["height"] - start_height) < 0.5:
                found_match = True
                floor_episode_data[floor_idx]["episodes"].append(episode)
                break
        if not found_match:
            floor_episode_data.append({"height": start_height, "episodes": [episode]})
            
    # Calculate the floor map and area for each floor
    for floor_idx, episode_data in enumerate(floor_episode_data):

        sim.set_agent_state(
            episode_data["episodes"][0]["start_position"],
            episode_data["episodes"][0]["start_rotation"],
        )
        floor_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
        number_of_navigable_cells = np.count_nonzero(floor_map)

        lower_bound, upper_bound = sim.pathfinder.get_bounds()
        grid_resolution = floor_map.shape
        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        floor_area = number_of_navigable_cells * np.product(grid_size)
        floor_episode_data[floor_idx]["floor_map"] = floor_map
        floor_episode_data[floor_idx]["floor_area"] = floor_area   

    
    scene_episodes = []
    episode_idx = 0
    for floor_idx, episode_data in enumerate(floor_episode_data):

        # Cluster starting points on each floor and pick cluster centroids
        num_clusters = assign_clusters_from_area_hm3d(episode_data["floor_area"])
        floor_points = np.array([ep["start_position"] for ep in episode_data["episodes"]]) # (N, 3)
        num_clusters = min(num_clusters, floor_points.shape[0])

        kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
        kmeans.fit(floor_points)
        
        # Find an episode closest to each cluster center
        init_floor_episodes = []
        for i in range(num_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            d2center = np.linalg.norm(floor_points - cluster_center[np.newaxis, :], axis=1)
            ep_idx = np.argmin(d2center).item()
            init_floor_episodes.append(episode_data["episodes"][ep_idx])

        if len(init_floor_episodes) < min_episode_threshold:
            continue

        # Calculate the geodesic distance between every episode start location
        X = [ep['start_position'] for ep in init_floor_episodes]
        dists = np.zeros((len(X), len(X))) + np.inf
        dists = [[sim.geodesic_distance(X[i], X[j]) for i in range(len(X))] for j in range(len(X))]
        dists = np.array(dists) # N x N

        # For each start position, generate a random sequence of N goals from nearby points
        floor_episodes = []
        for _ in range(num_resample): # x2 episodes per N
            for N in num_goals:
                for ep_idx in range(len(init_floor_episodes)):
                    nbhs = dists[ep_idx].argsort()[:N]
                    rs.shuffle(nbhs)
                    goals =  [init_floor_episodes[nbh]['start_position'] for nbh in nbhs.tolist()]
                    goals = [{'position': goal, 'radius': 0.2} for goal in goals]
                    episode = copy.deepcopy(init_floor_episodes[ep_idx])
                    episode['episode_id'] = episode_idx # replace original episode id
                    episode['goals'] = goals
                    episode_idx += 1
                    floor_episodes.append(episode)
        scene_episodes += floor_episodes
            
        if debug:
            visualize_episode_maps(floor_episodes, episode_data["floor_map"], sim, f'viz/floor_{floor_idx}.png')


    with gzip.open(f'{out_dir}/walkthroughs/content/{scene_key}.json.gz', "wt") as fp:
        json.dump({"episodes": scene_episodes}, fp)

    sim.close()


def generate_episodes(data_dir, out_dir, seed):

    os.makedirs(f'{out_dir}/walkthroughs/content', exist_ok=True)

    # Generate episodes for navigation agents: 
    # cluster room area into points of interest, generate episodes (navigation to sequence of points)
    scene_episodes = load_scene_episodes(data_dir, ['train', 'val'])
    Parallel(n_jobs=64, verbose=16)(delayed(generate)(episodes, out_dir, seed) for episodes in scene_episodes.values())

    # [Debug] Generate and save floormaps for episodes
    # generate(list(scene_episodes.values())[0], out_dir, seed, debug=True)

    # Aggregate all episodes and save
    episodes = []
    for fl in glob.glob(f'{out_dir}/walkthroughs/content/*.json.gz'):
        episodes += json.load(gzip.open(fl))['episodes']

    print (f'Saving {len(episodes)} episodes')
    with gzip.open(f'{out_dir}/walkthroughs.json.gz', "wt") as fp:
        json.dump({'episodes': episodes}, fp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_dir', default='data/datasets/pointnav/hm3d/v1')
    parser.add_argument('--out_dir', default='data/walkthrough_data/hm3d/v1')
    args = parser.parse_args()

    generate_episodes(args.data_dir, args.out_dir, args.seed)