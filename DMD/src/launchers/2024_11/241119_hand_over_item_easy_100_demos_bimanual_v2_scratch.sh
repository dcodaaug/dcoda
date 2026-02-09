#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --partition=slurm
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=/home/arthur/peract_bimanual/logs/slurm_logs/slurm_output_%j.out

python -m torch.distributed.launch --use_env --master_port=25682 scripts/mini_train.py \
    --num_gpus_per_node 1 \
    --batch_size 2 \
    --task handover_item \
    --sfm_method orbslam_bimanual \
    --instance_data_path ../instance-data/241119_hand_over_item_easy_100_demos_bimanual_v2_scratch \
    --colmap_data_folders ../instance-data/241119_hand_over_item_easy_100_demos_bimanual_v2_scratch/handover_item_easy_bimanual \
    --focal_lengths -110.85124795436963 \
    --max_epoch 30000 \
    --checkpoint_retention_interval 500 \
    --left_right_pose_cond \
    --wandb