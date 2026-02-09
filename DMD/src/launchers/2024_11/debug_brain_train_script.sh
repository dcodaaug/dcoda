#!/usr/bin/env bash
#SBATCH --gres=gpu:2
#SBATCH --partition=slurm
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=/home/arthur/peract_bimanual/logs/slurm_logs/slurm_output_%j.out

python -m torch.distributed.launch --use_env --master_port=25678 scripts/mini_train.py \
    --num_gpus_per_node 2 \
    --batch_size 2 \
    --task coordinated_lift_ball \
    --sfm_method orbslam_bimanual \
    --instance_data_path ../instance-data/241102_coordinated_lift_ball_100_demos_bimanual_v2_scratch \
    --colmap_data_folders ../instance-data/241102_coordinated_lift_ball_100_demos_bimanual_v2_scratch/coordinated_lift_ball_dmd_bimanual \
    --focal_lengths -110.85124795436964 \
    --max_epoch 20000 \
    --checkpoint_retention_interval 500 \
    --left_right_pose_cond \
    --wandb