# make sure you're in dcoda/ before executing the following command

python scripts/convert_dcoda_format_bimanual_D-Aug.py \
--input_dir=/home/zsh/dcoda/RLBench/tools/data/rlbench_data/coordinated_lift_ball/all_variations/episodes \
--output_dir=./DMD/instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual  \
--save_relative_poses \
--save_absolute_poses \
--save_depth_images \
--intervals=6