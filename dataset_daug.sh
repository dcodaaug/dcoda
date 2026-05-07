# make sure you're in dcoda/ before executing the following command

DISPLAY=:99 python scripts/daug_convert_to_rlbench_data_bimanual_v4.py \
--org_dir=/home/zsh/dcoda/data/rlbench_data/train/daug_co_lift_ball_smallgood/real100 \
--action_labels_dir=/home/zsh/dcoda/DMD/instance-data/260409_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1_action_labels \
--ik_debug \
--pose_smoothing_alpha=1.0 \
--left_joint_smoothing_alpha=1.0 \
--right_joint_smoothing_alpha=1.0 


