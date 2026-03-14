# make sure you're in dcoda/DMD/src/ before executing the following command

python scripts/annotate.py --sfm_method orbslam_bimanual \
--inf_json ../instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1/data.json \
--gt_json labels_10_bimanual.json --task_data_json labels_10_bimanual.json \
--output_dir ../instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1_action_labels \
--task_data_dir ../instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual