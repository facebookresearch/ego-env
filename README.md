# EgoEnv: Human-centric environment representations from egocentric video

This is the code accompanying our NeurIPS (oral) work:  
**EgoEnv: Human-centric environment representations from egocentric video**  
*Tushar Nagarajan, Santhosh Kumar Ramakrishnan, Ruta Desai, James Hillis, Kristen Grauman*  
[[arxiv]](https://arxiv.org/abs/2207.11365) [[project page]](https://vision.cs.utexas.edu/projects/ego-env/)

## Install
(1) Create a conda environment and install packages. This repo has been tested with python 3.8, torch 1.9 and cuda 10.2.
```
conda create -n egoenv python=3.8
```

(2) Install Pytorch
```
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

(3) Install Habitat 2.0 (sim + lab)
```
# Habitat sim
conda install habitat-sim=0.2.2 headless -c conda-forge -c aihabitat

# Habitat lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout e074ef86f0190e195e2929d4fa631c1aef30c0e1
pip install -e .
cd ..
```

(4) Install other required packages
```
pip install -r requirements.txt
```

## Download data and checkpoints
Download annotatations, sample training data and model checkpoints.
```
bash data/download_data.sh
```

## Generate environment features
Environment features can be generated using pre-trained model checkpoints. These models have been trained on simulated walkthroughs from Habitat environments. 

**Generate environment features** for a video at 1FPS and save to disk.
```
python generate_env_features.py \
    --config state_prediction/config/downstream.yaml \
    --video /path/to/video.mp4 \
    --save env_feats.pth \
    MODEL.WEIGHTS checkpoints/cardinal_object_state/lightning_logs/version_0/checkpoints/epoch=2279-val_loss=2.29E-01.ckpt
```

This generates a `(T, 128)` tensor for features sampled at 1FPS. See parameters in `state_prediction/config/downstream.yaml`. The following sections have instructions for generating simulated training data and training models from scratch.

## Generate simulated walkthroughs 

**(1) Download HM3D scene and pointnav data.** Follow the [official instructions](https://aihabitat.org/datasets/hm3d/) to download HM3D scenes. Pointnav data (`pointnav_hm3d_v1.zip`) can be downloaded from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). After extracting, the resulting file structure should look like this:
```
data
├── datasets/pointnav/hm3d/v1
│   └── train
│   ├── train_10_percent
│   ├── train_50_percent
│   └── val
└── scene_datasets
    └── hm3d
        ├── example 
        ├── hm3d_basis.scene_dataset_config.json 
        ├── minival 
        ├── train 
        └── val 
```

**(2) Generate episode configurations for navigation agents.** Episodes correspond to initial agent positions and the sequence of goal states that need to be visited. Output: `data/walkthrough_data/hm3d/v1/walkthroughs.json.gz`
```
python -m walkthrough_generation.generate_episodes
```

**(3) Run navigation agents to save walkthrough trajectory.** Trajectories consist of the camera-pose at each time-step for each trajectory. Output: `data/walkthrough_data/hm3d/v1/state/info/`
```
sbatch walkthrough_generation/generate_walkthrough.sh 
```

**(4) Create the dataset of training and validation episodes** uniformly sampled from all available scenes. Output: `data/walkthrough_data/hm3d/v1/episode_list.pth`
```
python -m walkthrough_generation.parse_agent_state --mode episodes
```

**(5) Generate metadata for each episode**. This includes RGB videos, frame features (ResNet50), object positions in all cardinal directions etc. Output: `data/walkthrough_data/hm3d/v1/state/[rgb|r50_feats|detected_objects]/`
```
sbatch walkthrough_generation/generate_agent_state.sh 
```

**(6) Consolidate metadata for pose embedding learning and local state prediction**. Output: `data/walkthrough_data/hm3d/v1/state/pose/` and `data/walkthrough_data/hm3d/v1/state/cardinal_object_state/`
```
python -m walkthrough_generation.parse_agent_state --mode pose
python -m walkthrough_generation.parse_agent_state --mode cardinal_object_state
```

This process will result in `.pth` files corresponding to the labels needed for the pre-training and fine-tuning stage in our framework. These include:
- `rgb` videos: walkthrough videos rendered at 5FPS
- `r50_feats`: ResNet-50 frame features for each time-step, used as input to the transformer encoder-decoder models.
- `pose (512, 3)`:  Agent camera pose (x, z, &theta;) at each timestep, used for pose-embedding learning.
- `objects (512, 4, 23)`: Binary tensor corresponding to the presense (or absense) of each object class in the four cardinal directions, at each timestep.
- `distances (512, 4, 23)`: Categorical labels for the discretized distance of each visible object in the cardinal directions (or -1 if the object is not present). 

Metadata for 50 sample trajectories are included in the [data download](#download-data-and-checkpoints), but the full dataset needs to be generated for training models from scratch.

## Local state prediction training

Training requires a single node (8 NVidia V100 GPUs). Once the walkthrough metadata is generated for all episodes, training occurs in two stages. 

**(1) Pre-train the pose embedding network.** Output: `checkpoints/pose_embed/.../epoch=####-val_loss=####.ckpt`
```
python -m state_prediction.train \
    --config state_prediction/config/pose_embed.yaml \
    CHECKPOINT_DIR checkpoints/pose_embed \
```

**(2) Train local state prediction models** using previously trained pose embeddings. Output: `checkpoints/cardinal_object_state/.../epoch=####-val_loss=####.ckpt`. 
```
python -m state_prediction.train \
    --config state_prediction/config/cardinal_objects.yaml \
    MODEL.POSE_MODEL_WEIGHTS /path/to/pose_embed/checkpoint.ckpt \
    CHECKPOINT_DIR checkpoints/cardinal_object_state
```

## Downstream video understanding tasks
Trained checkpoints are used downstream to [generate environment features](#generate-environment-features). Datasets for RoomPred and NLQ are in `data/annotations`. See [DATASETS.md](DATASETS.md) for more information. Pre-computed features for Ego4D NLQ videos can be downloaded [here](https://dl.fbaipublicfiles.com/ego-env/data/ego4d_egoenv_nlq_feats.zip).

## License
This project is released under the CC-BY-NC 4.0 license, as found in the LICENSE file.

## Cite
If you find this repository useful in your own research, please consider citing:
```
@inproceedings{nagarajan2023egoenv,
  title={EgoEnv: Human-centric environment representations from egocentric video},
  author={Nagarajan, Tushar and Ramakrishnan, Santhosh Kumar and Desai, Ruta and Hillis, James and Grauman, Kristen},
  booktitle={NeurIPS},
  year={2023}
}
```
