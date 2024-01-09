#!/bin/bash
#SBATCH --job-name=generate_agent_state
#SBATCH --array=0-158
#SBATCH --output=logs/generate_agent_state/log_%A_%a.out
#SBATCH --error=logs/generate_agent_state/log_%A_%a.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=110GB
#SBATCH --time=48:00:00
#SBATCH --partition=eht,learnlab,learnfair

GLOG_minloglevel=2 MAGNUM_LOG=quiet \
python -m walkthrough_generation.generate_agent_state \
    --data_dir data/walkthrough_data/hm3d/v1/ \
    --job_id ${SLURM_ARRAY_TASK_ID}