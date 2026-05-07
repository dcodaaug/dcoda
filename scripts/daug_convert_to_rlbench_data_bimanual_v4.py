import argparse
from os.path import dirname, join, abspath
from pyrep import PyRep
import os
import json
import pickle
import numpy as np
from pyrep.robots.arms.dual_panda import PandaLeft
from pyrep.robots.arms.dual_panda import PandaRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperLeft
from rlbench.backend.robot import BimanualRobot
from rlbench.demo import Demo
from scipy.spatial.transform import Rotation as R
import copy
import shutil
import time
from PIL import Image
from tqdm import tqdm
from helpers import sort_key

def from_blender_frame(transformation):
    pre_conversion = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1],
    ])
    
    transformation = np.asarray(transformation)
    transformation = np.matmul(transformation, np.linalg.inv(pre_conversion))
    transformation = np.matmul(pre_conversion, transformation)

    return transformation

def step_in_pyrep(pyrep):
    for i in range(20):
        pyrep.step()

def reset_bimanual(pyrep, robot, initial_robot_state, start_arm_joint_pos, starting_gripper_joint_pos):
    for arm, gripper in initial_robot_state:        
        pyrep.set_configuration_tree(arm)
        pyrep.set_configuration_tree(gripper)
    
    robot.right_arm.set_joint_positions(start_arm_joint_pos[0], disable_dynamics=True)
    robot.right_gripper.set_joint_positions(starting_gripper_joint_pos[0], disable_dynamics=True)

    robot.left_arm.set_joint_positions(start_arm_joint_pos[1], disable_dynamics=True)
    robot.left_gripper.set_joint_positions(starting_gripper_joint_pos[1], disable_dynamics=True)

def create_new_episode_dirs(new_dir, org_dir):
    os.makedirs(new_dir)
    shutil.copytree(os.path.join(org_dir, 'front_depth'), os.path.join(new_dir, 'front_depth'))
    shutil.copytree(os.path.join(org_dir, 'front_mask'), os.path.join(new_dir, 'front_mask'))
    shutil.copytree(os.path.join(org_dir, 'front_point_cloud'), os.path.join(new_dir, 'front_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'front_rgb'), os.path.join(new_dir, 'front_rgb'))

    shutil.copytree(os.path.join(org_dir, 'over_shoulder_left_depth'), os.path.join(new_dir, 'over_shoulder_left_depth'))
    shutil.copytree(os.path.join(org_dir, 'over_shoulder_left_mask'), os.path.join(new_dir, 'over_shoulder_left_mask'))
    shutil.copytree(os.path.join(org_dir, 'over_shoulder_left_point_cloud'), os.path.join(new_dir, 'over_shoulder_left_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'over_shoulder_left_rgb'), os.path.join(new_dir, 'over_shoulder_left_rgb'))

    shutil.copytree(os.path.join(org_dir, 'over_shoulder_right_depth'), os.path.join(new_dir, 'over_shoulder_right_depth'))
    shutil.copytree(os.path.join(org_dir, 'over_shoulder_right_mask'), os.path.join(new_dir, 'over_shoulder_right_mask'))
    shutil.copytree(os.path.join(org_dir, 'over_shoulder_right_point_cloud'), os.path.join(new_dir, 'over_shoulder_right_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'over_shoulder_right_rgb'), os.path.join(new_dir, 'over_shoulder_right_rgb'))

    shutil.copytree(os.path.join(org_dir, 'overhead_depth'), os.path.join(new_dir, 'overhead_depth'))
    shutil.copytree(os.path.join(org_dir, 'overhead_mask'), os.path.join(new_dir, 'overhead_mask'))
    shutil.copytree(os.path.join(org_dir, 'overhead_point_cloud'), os.path.join(new_dir, 'overhead_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'overhead_rgb'), os.path.join(new_dir, 'overhead_rgb'))

    os.makedirs(os.path.join(new_dir, 'wrist_left_depth'))
    os.makedirs(os.path.join(new_dir, 'wrist_left_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_point_cloud'), os.path.join(new_dir, 'wrist_left_point_cloud'))
    os.makedirs(os.path.join(new_dir, 'wrist_left_rgb'))

    os.makedirs(os.path.join(new_dir, 'wrist_right_depth'))
    os.makedirs(os.path.join(new_dir, 'wrist_right_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_right_point_cloud'), os.path.join(new_dir, 'wrist_right_point_cloud'))
    os.makedirs(os.path.join(new_dir, 'wrist_right_rgb'))

    shutil.copy(os.path.join(org_dir, 'variation_descriptions.pkl'), os.path.join(new_dir, 'variation_descriptions.pkl'))
    shutil.copy(os.path.join(org_dir, 'variation_number.pkl'), os.path.join(new_dir, 'variation_number.pkl'))

def save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, org_img_index, depth_npy):
    # save the left wrist original image to the new directory
    source_left_wrist_img_path = os.path.join(org_dir, 'wrist_left_rgb', f"rgb_{org_img_index:04}.png")
    dest_left_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_left_rgb', f"rgb_{saved_img_index:04}.png")
    shutil.copy(source_left_wrist_img_path, dest_left_wrist_img_path) 
    
    if depth_npy:
        source_left_wrist_img_path = os.path.join(org_dir, 'wrist_left_depth', f"depth_{org_img_index:04}.npy")
        dest_left_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_left_depth', f"depth_{saved_img_index:04}.npy")
    else:
        source_left_wrist_img_path = os.path.join(org_dir, 'wrist_left_depth', f"depth_{org_img_index:04}.png")
        dest_left_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_left_depth', f"depth_{saved_img_index:04}.png")
    shutil.copy(source_left_wrist_img_path, dest_left_wrist_img_path) 

    source_left_wrist_img_path = os.path.join(org_dir, 'wrist_left_mask', f"mask_{org_img_index:04}.png")
    dest_left_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_left_mask', f"mask_{saved_img_index:04}.png")
    shutil.copy(source_left_wrist_img_path, dest_left_wrist_img_path)

    # save the right wrist original image to the new directory
    source_right_wrist_img_path = os.path.join(org_dir, 'wrist_right_rgb', f"rgb_{org_img_index:04}.png")
    dest_right_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_right_rgb', f"rgb_{saved_img_index:04}.png")
    shutil.copy(source_right_wrist_img_path, dest_right_wrist_img_path)

    if depth_npy:
        source_right_wrist_img_path = os.path.join(org_dir, 'wrist_right_depth', f"depth_{org_img_index:04}.npy")
        dest_right_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_right_depth', f"depth_{saved_img_index:04}.npy")
    else:
        source_right_wrist_img_path = os.path.join(org_dir, 'wrist_right_depth', f"depth_{org_img_index:04}.png")
        dest_right_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_right_depth', f"depth_{saved_img_index:04}.png")
    shutil.copy(source_right_wrist_img_path, dest_right_wrist_img_path)

    source_right_wrist_img_path = os.path.join(org_dir, 'wrist_right_mask', f"mask_{org_img_index:04}.png")
    dest_right_wrist_img_path = os.path.join(new_dir_for_augmented_images, 'wrist_right_mask', f"mask_{saved_img_index:04}.png")
    shutil.copy(source_right_wrist_img_path, dest_right_wrist_img_path)

    # print('Added original observation at index:', saved_img_index)

def quaternion_to_rotation_matrix(q):
    """
    Convert a unit quaternion to a 3x3 rotation matrix.

    Parameters:
        q (array-like): Quaternion [x, y, z, w] or [qx, qy, qz, qw],
                        where w is the scalar part.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("Input quaternion must be a 4-element array [x, y, z, w]")

    x, y, z, w = q

    # Normalize the quaternion to ensure it's a unit quaternion
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        x /= norm
        y /= norm
        z /= norm
        w /= norm

    # Compute rotation matrix elements
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

    return R


def smooth_quaternion(prev_quat, curr_quat, alpha):
    prev_quat = np.asarray(prev_quat, dtype=np.float64)
    curr_quat = np.asarray(curr_quat, dtype=np.float64)

    # Keep interpolation on the shortest path in quaternion space.
    if np.dot(prev_quat, curr_quat) < 0:
        curr_quat = -curr_quat

    blended = (1 - alpha) * prev_quat + alpha * curr_quat
    norm = np.linalg.norm(blended)
    if norm < 1e-8:
        return curr_quat
    return blended / norm


def smooth_pose_matrix(curr_pose, prev_pose, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    curr_pose = np.asarray(curr_pose, dtype=np.float64)
    if prev_pose is None or alpha >= 1.0:
        return curr_pose.copy()

    prev_pose = np.asarray(prev_pose, dtype=np.float64)
    smoothed_pose = np.eye(4)
    smoothed_pose[:3, 3] = alpha * curr_pose[:3, 3] + (1 - alpha) * prev_pose[:3, 3]

    prev_quat = R.from_matrix(prev_pose[:3, :3]).as_quat()
    curr_quat = R.from_matrix(curr_pose[:3, :3]).as_quat()
    smoothed_quat = smooth_quaternion(prev_quat, curr_quat, alpha)
    smoothed_pose[:3, :3] = R.from_quat(smoothed_quat).as_matrix()
    return smoothed_pose


def smooth_joint_positions(curr_joint_positions, prev_joint_positions, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    curr_joint_positions = np.asarray(curr_joint_positions, dtype=np.float64)
    if prev_joint_positions is None or alpha >= 1.0:
        return curr_joint_positions.copy()

    prev_joint_positions = np.asarray(prev_joint_positions, dtype=np.float64)
    return alpha * curr_joint_positions + (1 - alpha) * prev_joint_positions

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='A simple script that greets the user.')

# Define an argument
parser.add_argument('--org_dir', type=str, required=True, help='Original data directory')
parser.add_argument('--action_labels_dir', type=str, required=True, help='Action labels directory')
parser.add_argument('--skip_inter_frames', action='store_true', help='Skip intermediate frames when using augmented state')
parser.add_argument('--filter', action='store_true', help='Filter out states with timesteps that exceed filter_t')
parser.add_argument('--filter_t', type=int, default=-1, help="Any states with timesteps greater than filter_t will be filtered out")
parser.add_argument('--depth_npy', action='store_true', help='Depth images are in .npy format')
parser.add_argument('--ik_debug', action='store_true', help='Print per-attempt IK diagnostics')
parser.add_argument('--pose_smoothing_alpha', type=float, default=0.8,
                    help='Pose smoothing factor in [0,1] before IK; 1.0 disables smoothing')
parser.add_argument('--left_joint_smoothing_alpha', type=float, default=0.4,
                    help='Left-arm joint smoothing factor in [0,1] after IK; 1.0 disables smoothing')
parser.add_argument('--right_joint_smoothing_alpha', type=float, default=0.4,
                    help='Right-arm joint smoothing factor in [0,1] after IK; 1.0 disables smoothing')

# Parse the arguments
args = parser.parse_args()

ep_folders = sorted(os.listdir(args.org_dir), key=sort_key)
traj_folders = sorted(os.listdir(args.action_labels_dir), key=sort_key)

assert len(ep_folders) == len(traj_folders), 'Number of episodes and action labels do not match'

pbar = tqdm(total=len(ep_folders))
total_ik_failures = {}

# Set up the PyRep environment
current_directory = os.getcwd()
DIR_PATH = os.path.join(current_directory, 'RLBench/rlbench')
headless = True
pyrep = PyRep()
pyrep.launch(join(DIR_PATH, 'task_design_bimanual.ttt'), headless=headless)
pyrep.start()

right_arm = PandaRight()
left_arm = PandaLeft()
right_gripper = PandaGripperRight()
left_gripper = PandaGripperLeft()
robot = BimanualRobot(right_arm, right_gripper, left_arm, left_gripper)
initial_robot_state = [(robot.right_arm.get_configuration_tree(),
                        robot.right_gripper.get_configuration_tree()),
                        (robot.left_arm.get_configuration_tree(),
                        robot.left_gripper.get_configuration_tree())]
start_arm_joint_pos = [robot.right_arm.get_joint_positions(), robot.left_arm.get_joint_positions()]
starting_gripper_joint_pos = [robot.right_gripper.get_joint_positions(), robot.left_gripper.get_joint_positions()]


# loop through the episode folders
for i, ep_folder in enumerate(ep_folders):
    # open low_dim_obs pickle file
    curr_folder = f'episode{i}'
    with open(os.path.join(args.org_dir, curr_folder, 'low_dim_obs.pkl'), 'rb') as file:
        low_dim_obs = pickle.load(file)

    # current traj data
    cur_traj_img_folder = os.path.join(args.action_labels_dir, traj_folders[i], 'images')
    cur_traj_json = os.path.join(args.action_labels_dir, traj_folders[i], 'labels_10_bimanual_augment.json')

    # Open and read the JSON file
    with open(cur_traj_json, 'r') as file:
        cur_traj_json = json.load(file)

    # create a new dictionary for the augmented images
    new_dir_for_augmented_images = os.path.join(args.org_dir, f'episode{i+len(ep_folders)}')
    org_dir = os.path.join(args.org_dir, curr_folder)
    create_new_episode_dirs(new_dir_for_augmented_images, org_dir)

    w_l_filtered_cur_traj_json = {key: value for key, value in cur_traj_json.items() if 'w_l_diffusion_' in key}
    w_r_filtered_cur_traj_json = {key: value for key, value in cur_traj_json.items() if 'w_r_diffusion_' in key}

    # initialize some variables
    total_timesteps = len(low_dim_obs)
    new_dir_wrist_left_rgb_dir = os.path.join(new_dir_for_augmented_images, 'wrist_left_rgb')
    new_dir_wrist_right_rgb_dir = os.path.join(new_dir_for_augmented_images, 'wrist_right_rgb')

    bimanual_observations = []
    ik_failures = 0
    saved_img_index = 0
    prev_smoothed_left_wrist_cam_aug_pose = None
    prev_smoothed_right_wrist_cam_aug_pose = None
    prev_smoothed_left_joint_positions = None
    prev_smoothed_right_joint_positions = None
    prev_output_left_joint_positions = None
    prev_output_right_joint_positions = None
    left_aug_by_idx = {}
    right_aug_by_idx = {}
    for w_l_aug_image_name, w_l_aug_image_data in w_l_filtered_cur_traj_json.items():
        idx = int(w_l_aug_image_data[3].split('.')[0].split('_')[-1])
        left_aug_by_idx[idx] = (w_l_aug_image_name, w_l_aug_image_data)
    for w_r_aug_image_name, w_r_aug_image_data in w_r_filtered_cur_traj_json.items():
        idx = int(w_r_aug_image_data[3].split('.')[0].split('_')[-1])
        right_aug_by_idx[idx] = (w_r_aug_image_name, w_r_aug_image_data)

    assert set(left_aug_by_idx.keys()) == set(right_aug_by_idx.keys()), 'Left and right augmented frame indices do not match'
    aug_indices = sorted(set(left_aug_by_idx.keys()) & set(right_aug_by_idx.keys()), reverse=True)
    aug_indices = [idx for idx in aug_indices if 0 <= idx < total_timesteps]

    augmented_obs_by_idx = {}
    augmented_img_names_by_idx = {}

    if len(aug_indices) > 0:
        seed_index = 50 if total_timesteps > 50 else min(max(aug_indices) + 1, total_timesteps - 1)
        prev_output_left_joint_positions = np.asarray(low_dim_obs[seed_index].left.joint_positions, dtype=np.float64).copy()
        prev_output_right_joint_positions = np.asarray(low_dim_obs[seed_index].right.joint_positions, dtype=np.float64).copy()

    # Reverse solve 49->0, seeded by real step50 joints for better stitching with following original frames.
    for original_image_index in aug_indices:
        w_l_aug_image_name, w_l_aug_image_data = left_aug_by_idx[original_image_index]
        w_r_aug_image_name, w_r_aug_image_data = right_aug_by_idx[original_image_index]

        #################### left arm ####################
        left_arm_joint_positions = prev_output_left_joint_positions.copy()
        right_arm_joint_positions = prev_output_right_joint_positions.copy()
        left_arm_left_wrist_cam_pose = low_dim_obs[original_image_index].perception_data['wrist_left_pose']

        robot.left_arm.set_joint_positions(left_arm_joint_positions, disable_dynamics=True)
        robot.right_arm.set_joint_positions(right_arm_joint_positions, disable_dynamics=True)
        step_in_pyrep(pyrep)

        left_wrist_aug_transform = np.eye(4)
        left_wrist_aug_transform[:3,:3] = w_l_aug_image_data[6]
        left_wrist_aug_transform[:3,3] = w_l_aug_image_data[5]
        left_wrist_aug_transform = from_blender_frame(left_wrist_aug_transform)

        left_arm_left_wrist_cam_aug_pose = np.dot(left_wrist_aug_transform, left_arm_left_wrist_cam_pose)
        left_arm_left_wrist_cam_aug_pose = smooth_pose_matrix(
            left_arm_left_wrist_cam_aug_pose,
            prev_smoothed_left_wrist_cam_aug_pose,
            args.pose_smoothing_alpha,
        )
        prev_smoothed_left_wrist_cam_aug_pose = left_arm_left_wrist_cam_aug_pose.copy()

        t_left_wrist_cam_to_left_eff = np.dot(np.linalg.inv(low_dim_obs[original_image_index].perception_data['wrist_left_pose']), low_dim_obs[original_image_index].left.gripper_matrix)
        left_eff_frame = np.dot(left_arm_left_wrist_cam_aug_pose, t_left_wrist_cam_to_left_eff)

        left_x, left_y, left_z = left_eff_frame[:3,3]
        left_rotation_matrix_eff_frame = left_eff_frame[:3,:3]
        left_rotation = R.from_matrix(left_rotation_matrix_eff_frame)
        left_quaternion = left_rotation.as_quat()

        left_ik_succeeded = True
        try:
            left_arm_new_joint_positions = robot.left_arm.solve_ik_via_sampling([left_x, left_y, left_z], quaternion=left_quaternion, ignore_collisions=True, distance_threshold=1.4, max_configs=30, trials=100)[0]
        except:
            print('!!!!!!!!!! IK failed for left augmented image at index:', original_image_index)
            ik_failures += 1
            left_ik_succeeded = False
            left_arm_new_joint_positions = np.asarray(
                low_dim_obs[original_image_index].left.joint_positions,
                dtype=np.float64,
            ).copy()

        if left_ik_succeeded:
            left_arm_new_joint_positions = smooth_joint_positions(
                left_arm_new_joint_positions,
                prev_smoothed_left_joint_positions,
                args.left_joint_smoothing_alpha,
            )
        prev_smoothed_left_joint_positions = left_arm_new_joint_positions.copy()

        robot.left_arm.set_joint_positions(left_arm_new_joint_positions, disable_dynamics=True)
        left_arm_new_gripper_pose = [left_x, left_y, left_z, *left_quaternion]
        left_arm_new_gripper_matrix = np.eye(4)
        left_arm_new_gripper_matrix[:3,:3] = left_rotation_matrix_eff_frame
        left_arm_new_gripper_matrix[:3,3] = [left_x, left_y, left_z]

        #################### right arm ####################
        robot.left_arm.set_joint_positions(left_arm_joint_positions, disable_dynamics=True)
        robot.right_arm.set_joint_positions(right_arm_joint_positions, disable_dynamics=True)
        step_in_pyrep(pyrep)

        right_arm_right_wrist_cam_pose = low_dim_obs[original_image_index].perception_data['wrist_right_pose']

        right_wrist_aug_transform = np.eye(4)
        right_wrist_aug_transform[:3,:3] = w_r_aug_image_data[6]
        right_wrist_aug_transform[:3,3] = w_r_aug_image_data[5]
        right_wrist_aug_transform = from_blender_frame(right_wrist_aug_transform)

        right_arm_right_wrist_cam_aug_pose = np.dot(right_wrist_aug_transform, right_arm_right_wrist_cam_pose)
        right_arm_right_wrist_cam_aug_pose = smooth_pose_matrix(
            right_arm_right_wrist_cam_aug_pose,
            prev_smoothed_right_wrist_cam_aug_pose,
            args.pose_smoothing_alpha,
        )
        prev_smoothed_right_wrist_cam_aug_pose = right_arm_right_wrist_cam_aug_pose.copy()

        t_right_wrist_cam_to_right_eff = np.dot(np.linalg.inv(low_dim_obs[original_image_index].perception_data['wrist_right_pose']), low_dim_obs[original_image_index].right.gripper_matrix)
        right_eff_frame = np.dot(right_arm_right_wrist_cam_aug_pose, t_right_wrist_cam_to_right_eff)

        right_x, right_y, right_z = right_eff_frame[:3,3]
        right_rotation_matrix_eff_frame = right_eff_frame[:3,:3]
        right_rotation = R.from_matrix(right_rotation_matrix_eff_frame)
        right_quaternion = right_rotation.as_quat()

        right_ik_succeeded = True
        try:
            right_arm_new_joint_positions = robot.right_arm.solve_ik_via_sampling([right_x, right_y, right_z], quaternion=right_quaternion, ignore_collisions=True, distance_threshold=1.4, max_configs=30, trials=100)[0]
        except:
            print('!!!!!!!!!! IK failed for right augmented image at index:', original_image_index)
            ik_failures += 1
            right_ik_succeeded = False
            right_arm_new_joint_positions = np.asarray(
                low_dim_obs[original_image_index].right.joint_positions,
                dtype=np.float64,
            ).copy()

        if right_ik_succeeded:
            right_arm_new_joint_positions = smooth_joint_positions(
                right_arm_new_joint_positions,
                prev_smoothed_right_joint_positions,
                args.right_joint_smoothing_alpha,
            )
        prev_smoothed_right_joint_positions = right_arm_new_joint_positions.copy()

        robot.right_arm.set_joint_positions(right_arm_new_joint_positions, disable_dynamics=True)
        right_arm_new_gripper_pose = [right_x, right_y, right_z, *right_quaternion]
        right_arm_new_gripper_matrix = np.eye(4)
        right_arm_new_gripper_matrix[:3,:3] = right_rotation_matrix_eff_frame
        right_arm_new_gripper_matrix[:3,3] = [right_x, right_y, right_z]

        bimanual_obs = copy.deepcopy(low_dim_obs[original_image_index])
        bimanual_obs.left.joint_positions = left_arm_new_joint_positions
        bimanual_obs.left.gripper_pose = left_arm_new_gripper_pose
        bimanual_obs.left.gripper_matrix = left_arm_new_gripper_matrix
        bimanual_obs.right.joint_positions = right_arm_new_joint_positions
        bimanual_obs.right.gripper_pose = right_arm_new_gripper_pose
        bimanual_obs.right.gripper_matrix = right_arm_new_gripper_matrix
        bimanual_obs.perception_data['wrist_left_pose'] = left_arm_left_wrist_cam_aug_pose
        bimanual_obs.perception_data['wrist_right_pose'] = right_arm_right_wrist_cam_aug_pose

        augmented_obs_by_idx[original_image_index] = bimanual_obs
        augmented_img_names_by_idx[original_image_index] = (w_l_aug_image_name, w_r_aug_image_name)

        prev_output_left_joint_positions = np.asarray(left_arm_new_joint_positions, dtype=np.float64).copy()
        prev_output_right_joint_positions = np.asarray(right_arm_new_joint_positions, dtype=np.float64).copy()

    total_ik_failures[curr_folder] = ik_failures

    # Write final trajectory in chronological order.
    for j in range(total_timesteps):
        if j in augmented_obs_by_idx:
            bimanual_observations.append(augmented_obs_by_idx[j])

            w_l_aug_image_name, w_r_aug_image_name = augmented_img_names_by_idx[j]
            dest_aug_left_wrist_img_path = os.path.join(new_dir_wrist_left_rgb_dir, f"rgb_{saved_img_index:04}.png")
            aug_left_wrist_img = Image.open(os.path.join(args.action_labels_dir, traj_folders[i], 'images', w_l_aug_image_name))
            aug_left_wrist_img = aug_left_wrist_img.resize((128, 128))
            aug_left_wrist_img.save(dest_aug_left_wrist_img_path)

            dest_aug_right_wrist_img_path = os.path.join(new_dir_wrist_right_rgb_dir, f"rgb_{saved_img_index:04}.png")
            aug_right_wrist_img = Image.open(os.path.join(args.action_labels_dir, traj_folders[i], 'images', w_r_aug_image_name))
            aug_right_wrist_img = aug_right_wrist_img.resize((128, 128))
            aug_right_wrist_img.save(dest_aug_right_wrist_img_path)
        else:
            local_bimanual_obs = copy.deepcopy(low_dim_obs[j])
            bimanual_observations.append(local_bimanual_obs)
            save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, j, depth_npy=args.depth_npy)

        saved_img_index += 1

    # NOTE: here, we assume the new demo has the same number of observations as the original demo
    new_demo = Demo(bimanual_observations, low_dim_obs.random_seed)
    # save new_demo to a pickle file
    with open(os.path.join(new_dir_for_augmented_images, 'low_dim_obs.pkl'), 'wb') as f:
        pickle.dump(new_demo, f)
    reset_bimanual(pyrep, robot, initial_robot_state, start_arm_joint_pos, starting_gripper_joint_pos)
    pbar.update(1)

pyrep.stop()
pyrep.shutdown()
pbar.close()
print('Finished generating augmented data!')
print('total_ik_failures: ', total_ik_failures)