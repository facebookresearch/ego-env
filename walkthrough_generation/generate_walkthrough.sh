#!/bin/bash
#SBATCH --job-name=generate_walkthrough
#SBATCH --array=0-984
#SBATCH --output=logs/generate_walkthrough/log_%A_%a.out
#SBATCH --error=logs/generate_walkthrough/log_%A_%a.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=40GB
#SBATCH --time=4:00:00
#SBATCH --partition=eht,learnlab,learnfair

python -m walkthrough_generation.generate_walkthrough \
    --walkthrough_dir data/datasets/pointnav/hm3d/v1/walkthroughs/content/ \
    --exp_config walkthrough_generation/config/planner.yaml \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    TASK_CONFIG.DATASET.DATA_PATH data/walkthrough_data/hm3d/v1/walkthroughs.json.gz \
    EVAL.SPLIT walkthroughs \
    VIDEO_DIR data/walkthrough_data/hm3d/v1/state/ \
    LOG_FILE data/walkthrough_data/hm3d/v1/state/generation_logs.txt \
    TASK_CONFIG.TASK.TYPE OracleExp-v0