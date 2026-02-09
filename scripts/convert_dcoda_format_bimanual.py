import argparse
import os
from PIL import Image
import pickle
import json
import numpy as np
from helpers import sort_key

def save_images_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path):
    wrist_left_rgb_folder = os.path.join(ep_folder_path, 'wrist_left_rgb')
    wrist_left_rgb_files = sorted(os.listdir(wrist_left_rgb_folder), key=sort_key)
    for i, rgb_file in enumerate(wrist_left_rgb_files):
        rgb_file_path = os.path.join(wrist_left_rgb_folder, rgb_file)

        # Load the PNG image
        png_image = Image.open(rgb_file_path)

        # Convert to RGB (JPEG doesn't support transparency/alpha channel)
        rgb_image = png_image.convert('RGB')

        image_filename = f"wrist_left_{i:05d}.jpg"

        output_image_file_path = os.path.join(dcoda_traj_images_folder, image_filename) # The format {i:05d} pads the number i with leading zeros to 5 digits

        # Save the image as a .jpg file
        rgb_image.save(output_image_file_path)

    wrist_right_rgb_folder = os.path.join(ep_folder_path, 'wrist_right_rgb')
    wrist_right_rgb_files = sorted(os.listdir(wrist_right_rgb_folder), key=sort_key)
    for i, rgb_file in enumerate(wrist_right_rgb_files):
        rgb_file_path = os.path.join(wrist_right_rgb_folder, rgb_file)

        # Load the PNG image
        png_image = Image.open(rgb_file_path)

        # Convert to RGB (JPEG doesn't support transparency/alpha channel)
        rgb_image = png_image.convert('RGB')

        image_filename = f"wrist_right_{i:05d}.jpg"

        output_image_file_path = os.path.join(dcoda_traj_images_folder, image_filename) # The format {i:05d} pads the number i with leading zeros to 5 digits

        # Save the image as a .jpg file
        rgb_image.save(output_image_file_path)

    print(f'Finished saving RGB images to {dcoda_traj_images_folder}')

def save_absolute_poses_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path, low_dim_obs):
    """
    Adapted from the "to_absolute_pose_json" function from ErinZhang1998/dmd_diffusion/data/generate_slam_labels.py
    """
    raw_labels = {}
    gripper_labels = {}
    gripper_matrices = {}

    wrist_left_rgb_folder = os.path.join(ep_folder_path, 'wrist_left_rgb')
    wrist_left_rgb_files = sorted(os.listdir(wrist_left_rgb_folder), key=sort_key)
    wrist_right_rgb_folder = os.path.join(ep_folder_path, 'wrist_right_rgb')
    wrist_right_rgb_files = sorted(os.listdir(wrist_right_rgb_folder), key=sort_key)

    i = 0
    for wrist_left_rgb_file, wrist_right_rgb_file in zip(wrist_left_rgb_files, wrist_right_rgb_files):
        wrist_left_image_filename = f"wrist_left_{i:05d}.jpg"
        raw_labels[wrist_left_image_filename] = low_dim_obs[i].perception_data['wrist_left_pose'].tolist()
        wrist_right_image_filename = f"wrist_right_{i:05d}.jpg"
        raw_labels[wrist_right_image_filename] = low_dim_obs[i].perception_data['wrist_right_pose'].tolist()
        gripper_filename = f"{i:05d}"        
        gripper_labels[gripper_filename] = [low_dim_obs[i].left.gripper_open, low_dim_obs[i].right.gripper_open]
        gripper_matrices[wrist_left_image_filename] = low_dim_obs[i].left.gripper_matrix.tolist()
        gripper_matrices[wrist_right_image_filename] = low_dim_obs[i].right.gripper_matrix.tolist()
        i += 1

    # save raw_labels.json
    raw_labels_filepath = os.path.join('/'.join(dcoda_traj_images_folder.split('/')[:-1]), 'raw_labels_bimanual.json')
    with open(raw_labels_filepath, 'w') as json_file:
        json.dump(raw_labels, json_file, indent=4)  # indent=4 makes the file more readable
    print(f'Finished saving {raw_labels_filepath}')

    # save gripper_state
    gripper_state_filepath = os.path.join('/'.join(dcoda_traj_images_folder.split('/')[:-1]), 'gripper_state_bimanual.json')
    with open(gripper_state_filepath, 'w') as json_file:
        json.dump(gripper_labels, json_file, indent=4)  # indent=4 makes the file more readable
    print(f'Finished saving {gripper_state_filepath}')

    # save gripper matrices
    gripper_matrices_filepath = os.path.join('/'.join(dcoda_traj_images_folder.split('/')[:-1]), 'gripper_matrices.json')
    with open(gripper_matrices_filepath, 'w') as json_file:
        json.dump(gripper_matrices, json_file, indent=4)  # indent=4 makes the file more readable
    print(f'Finished saving {gripper_matrices_filepath}')

def save_relative_poses_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path, low_dim_obs, intervals):
    """
    Adapted from the "to_relative_pose_json" function from ErinZhang1998/dmd_diffusion/data/generate_slam_labels.py
    """
    labels_10_bimanual = {}

    wrist_left_rgb_folder = os.path.join(ep_folder_path, 'wrist_left_rgb')
    rgb_files = sorted(os.listdir(wrist_left_rgb_folder), key=sort_key)
    start_frame = 0
    end_frame = len(rgb_files)

    # process data for raw_labels based on the to_relative_pose_json function
    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        goal_frame = frame_idx + intervals
        if goal_frame >= end_frame:
            break

        w_l_cam_pose1 = low_dim_obs[i].perception_data['wrist_left_pose']
        w_l_cam_pose2 = low_dim_obs[i+intervals].perception_data['wrist_left_pose']

        T_w_l_cam1_cam2 = np.dot(np.linalg.inv(w_l_cam_pose1), w_l_cam_pose2)
        T_w_l_cam1_cam2_t = T_w_l_cam1_cam2[:3,-1].astype(float)
        T_w_l_cam1_cam2_r = T_w_l_cam1_cam2[:3,:3].tolist()

        w_l_frame_name = f"wrist_left_{frame_idx:05}.jpg"
        w_l_frame_namelk = f"wrist_left_{goal_frame:05}.jpg"
        labels_10_bimanual[w_l_frame_name] = [list(T_w_l_cam1_cam2_t), T_w_l_cam1_cam2_r, w_l_frame_namelk]

        w_r_cam_pose1 = low_dim_obs[i].perception_data['wrist_right_pose']
        w_r_cam_pose2 = low_dim_obs[i+intervals].perception_data['wrist_right_pose']

        T_w_r_cam1_cam2 = np.dot(np.linalg.inv(w_r_cam_pose1), w_r_cam_pose2)
        T_w_r_cam1_cam2_t = T_w_r_cam1_cam2[:3,-1].astype(float)
        T_w_r_cam1_cam2_r = T_w_r_cam1_cam2[:3,:3].tolist()

        w_r_frame_name = f"wrist_right_{frame_idx:05}.jpg"
        w_r_frame_namelk = f"wrist_right_{goal_frame:05}.jpg"
        labels_10_bimanual[w_r_frame_name] = [list(T_w_r_cam1_cam2_t), T_w_r_cam1_cam2_r, w_r_frame_namelk]

    # save labels_10.json
    labels_10_filepath = os.path.join('/'.join(dcoda_traj_images_folder.split('/')[:-1]), 'labels_10_bimanual.json')
    with open(labels_10_filepath, 'w') as json_file:
        json.dump(labels_10_bimanual, json_file, indent=4)  # indent=4 makes the file more readable
    print(f'Finished saving {labels_10_filepath}')
    print('wrist left focal length: ', low_dim_obs[0].misc['wrist_left_camera_intrinsics'][0][0])
    print('wrist right focal length: ', low_dim_obs[0].misc['wrist_right_camera_intrinsics'][0][0])

def save_depth_images_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path, low_dim_obs, save_depth_npy):
    wrist_left_depth_folder = os.path.join(ep_folder_path, 'wrist_left_depth')
    wrist_left_depth_files = sorted(os.listdir(wrist_left_depth_folder), key=sort_key)
    depth_info = {}
    for i, depth_file in enumerate(wrist_left_depth_files):
        depth_file_path = os.path.join(wrist_left_depth_folder, depth_file)

        if save_depth_npy:
            depth_file_path = depth_file_path.replace('wrist_left_depth', 'wrist_left_depth_np')
            depth_file_path = depth_file_path.replace('.png', '.npy')
            depth_image = np.load(depth_file_path, allow_pickle=True)

            image_filename = f"wrist_left_{i:05d}_depth.npy"

            output_image_file_path = os.path.join(dcoda_traj_images_folder, image_filename)

            np.save(output_image_file_path, depth_image)
        else:
            # Load the PNG image
            png_image = Image.open(depth_file_path)

            image_filename = f"wrist_left_{i:05d}_depth.png"

            output_image_file_path = os.path.join(dcoda_traj_images_folder, image_filename)

            # Save the image as a .jpg file
            png_image.save(output_image_file_path)

        # save depth info
        depth_info[image_filename] = {
            'wrist_left_camera_near': low_dim_obs[i].misc['wrist_left_camera_near'],
            'wrist_left_camera_far': low_dim_obs[i].misc['wrist_left_camera_far'],
        }

    wrist_right_depth_folder = os.path.join(ep_folder_path, 'wrist_right_depth')
    wrist_right_depth_files = sorted(os.listdir(wrist_right_depth_folder), key=sort_key)
    for i, depth_file in enumerate(wrist_right_depth_files):
        depth_file_path = os.path.join(wrist_right_depth_folder, depth_file)

        if save_depth_npy:
            depth_file_path = depth_file_path.replace('wrist_right_depth', 'wrist_right_depth_np')
            depth_file_path = depth_file_path.replace('.png', '.npy')
            depth_image = np.load(depth_file_path, allow_pickle=True)

            image_filename = f"wrist_right_{i:05d}_depth.npy"

            output_image_file_path = os.path.join(dcoda_traj_images_folder, image_filename)

            np.save(output_image_file_path, depth_image)
        else:
            # Load the PNG image
            png_image = Image.open(depth_file_path)

            image_filename = f"wrist_right_{i:05d}_depth.png"

            output_image_file_path = os.path.join(dcoda_traj_images_folder, image_filename)

            # Save the image as a .jpg file
            png_image.save(output_image_file_path)

        # save depth info
        depth_info[image_filename] = {
            'wrist_right_camera_near': low_dim_obs[i].misc['wrist_right_camera_near'],
            'wrist_right_camera_far': low_dim_obs[i].misc['wrist_right_camera_far'],
        }
    
    # save depth_info.json
    depth_info_filepath = os.path.join('/'.join(dcoda_traj_images_folder.split('/')[:-1]), 'depth_info.json')
    with open(depth_info_filepath, 'w') as json_file:
        json.dump(depth_info, json_file, indent=4)  # indent=4 makes the file more readable

    print(f'Finished saving depth images to {dcoda_traj_images_folder}')

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Convert data to dcoda format')

# Define an argument
parser.add_argument('--input_dir', type=str, help='Input data directory')
parser.add_argument("--input_dirs", nargs="+", type=str, help="Input data directories")
parser.add_argument('--output_dir', type=str, required=True, help='Output data directory')
parser.add_argument('--save_relative_poses',action='store_true')
parser.add_argument('--save_absolute_poses',action='store_true')
parser.add_argument('--save_depth_images',action='store_true')
parser.add_argument('--save_depth_npy',action='store_true')
parser.add_argument('--intervals', type=int, default=12, help="intervals to calculate relative poses")

# Parse the arguments
args = parser.parse_args()

list_of_dirs = args.input_dirs
if list_of_dirs is None or len(list_of_dirs) == 0:
    list_of_dirs = [args.input_dir]

# Create the output directory, including any necessary intermediate directories
os.makedirs(args.output_dir, exist_ok=True)

ep_num = 0
for input_dir in list_of_dirs:
    # List and sort the directory contents
    ep_folders = sorted(os.listdir(input_dir), key=sort_key)

    # loop through the episode folders
    for i, ep_folder in enumerate(ep_folders):
        # ep_num = int(ep_folder.split('episode')[-1])
        dcoda_traj_folder = os.path.join(args.output_dir, f'traj{ep_num}')

        # create images folder
        dcoda_traj_images_folder = os.path.join(dcoda_traj_folder, 'images')
        os.makedirs(dcoda_traj_images_folder, exist_ok=True)

        # open low_dim_obs pickle file
        with open(os.path.join(input_dir, ep_folder, 'low_dim_obs.pkl'), 'rb') as file:
            low_dim_obs = pickle.load(file)
    
        ep_folder_path = os.path.join(input_dir, ep_folder)

        if args.save_relative_poses:
            save_relative_poses_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path, low_dim_obs, args.intervals)
        if args.save_absolute_poses:
            save_absolute_poses_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path, low_dim_obs)
        if args.save_depth_images:
            save_depth_images_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path, low_dim_obs, args.save_depth_npy)
        save_images_to_dcoda_folder(dcoda_traj_images_folder, ep_folder_path)
        ep_num += 1
