# make sure you're in dcoda/ before executing the following command

DISPLAY=:99 python scripts/convert_to_rlbench_data_bimanual_v4.py \
--org_dir=./data/rlbench_data/train/coordinated_lift_ball_200_demos_128x128_daug/coordinated_lift_ball/all_variations/episodes \
--action_labels_dir=./DMD/instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1_action_labels 