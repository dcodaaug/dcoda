# make sure you're in dcoda/DMD/src/ before executing the following commands
# --task coordinated_lift_ball \
# DISPLAY=:99 python scripts/generate_inference_json_vlms.py \
# --sfm_method orbslam_bimanual \
# --data_folders ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
# --focal_lengths -110.85124795436963 --image_heights 128 \
# --suffix v1 \
# --output_root ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
# --sample_rotation --every_x_frame 6 --mult 1 --seed 10 --save_robot_traj_visualization --save_print_to_file


python scripts/sample-imgs-multi.py ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1/data.json \
--model /nas/datasets/zsh/MVDA_ckpts/dmd_co_lift_ball/checkpoints/17000-00000000.pth \
--sfm_method orbslam_bimanual \
--image_size 128 --batch_size 64 --gpus 0,1,2