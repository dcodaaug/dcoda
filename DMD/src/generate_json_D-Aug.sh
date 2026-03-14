DISPLAY=:99 python scripts/generate_inference_json_vlms_D-Aug.py \
--sfm_method orbslam_bimanual \
--data_folders ../instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
--focal_lengths -110.85124795436963 --image_heights 128 \
--suffix v1 \
--output_root ../instance-data/260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
--sample_rotation --every_x_frame 1 --mult 1 --seed 10 --save_robot_traj_visualization --save_print_to_file \
--cutoff_num 50 \
--no_opt \
--visualize_interval 3