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
from PIL import Image
from tqdm import tqdm
from helpers import sort_key

DISTANCE_BETWEEN_LEFT_RIGHT_ARMS = 1.22 # meters
REAL_LEFT_ARM_STARTING_STATE = [1.3629151365049208, -1.1219412063110294, -2.036740639010723, -1.632770313019793, 1.7637373624131287, 2.771111558000195]
REAL_RIGHT_ARM_STARTING_STATE = [-1.4106484960522758, -1.9944810024925443, 2.054192150410259, -1.5718085742732786, -1.6140418455715997, 0.061382133371988376]
ENV_SWIFT_DT = 0.05

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

def reset_bimanual_real_robot(env_swift, left_robot, right_robot):
    left_robot.q = REAL_LEFT_ARM_STARTING_STATE
    right_robot.q = REAL_RIGHT_ARM_STARTING_STATE
    env_swift.step(ENV_SWIFT_DT)

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

def create_new_episode_dirs_real(new_dir, org_dir):
    os.makedirs(new_dir)
    # creating the front, over_shoulder_left, over_shoulder_right, and overhead directories, but we don't use them on the real robot
    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'front_depth'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_mask'), os.path.join(new_dir, 'front_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'front_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_rgb'), os.path.join(new_dir, 'front_rgb'))

    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'over_shoulder_left_depth'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_mask'), os.path.join(new_dir, 'over_shoulder_left_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'over_shoulder_left_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_rgb'), os.path.join(new_dir, 'over_shoulder_left_rgb'))

    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'over_shoulder_right_depth'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_mask'), os.path.join(new_dir, 'over_shoulder_right_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'over_shoulder_right_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_rgb'), os.path.join(new_dir, 'over_shoulder_right_rgb'))

    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'overhead_depth'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_mask'), os.path.join(new_dir, 'overhead_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'overhead_point_cloud'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_rgb'), os.path.join(new_dir, 'overhead_rgb'))

    os.makedirs(os.path.join(new_dir, 'wrist_left_depth'))
    os.makedirs(os.path.join(new_dir, 'wrist_left_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_left_depth'), os.path.join(new_dir, 'wrist_left_point_cloud'))
    os.makedirs(os.path.join(new_dir, 'wrist_left_rgb'))

    os.makedirs(os.path.join(new_dir, 'wrist_right_depth'))
    os.makedirs(os.path.join(new_dir, 'wrist_right_mask'))
    shutil.copytree(os.path.join(org_dir, 'wrist_right_depth'), os.path.join(new_dir, 'wrist_right_point_cloud'))
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

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='A simple script that greets the user.')

# Define an argument
parser.add_argument('--org_dir', type=str, required=True, help='Original data directory')
parser.add_argument('--action_labels_dir', type=str, required=True, help='Action labels directory')
parser.add_argument('--skip_inter_frames', action='store_true', help='Skip intermediate frames when using augmented state')
parser.add_argument('--filter', action='store_true', help='Filter out states with timesteps that exceed filter_t')
parser.add_argument('--filter_t', type=int, default=-1, help="Any states with timesteps greater than filter_t will be filtered out")
parser.add_argument('--depth_npy', action='store_true', help='Depth images are in .npy format')
parser.add_argument("--real_robot", action="store_true", help="Whether to do inference on real robot data")

# Parse the arguments
args = parser.parse_args()

ep_folders = sorted(os.listdir(args.org_dir), key=sort_key)
traj_folders = sorted(os.listdir(args.action_labels_dir), key=sort_key)

assert len(ep_folders) == len(traj_folders), 'Number of episodes and action labels do not match'

pbar = tqdm(total=len(ep_folders))
total_ik_failures = {}

if args.real_robot:
    import swift
    import roboticstoolbox as rtb
    import spatialmath as sm

    env_swift = swift.Swift()
    env_swift.launch(realtime=True)
    left_robot = rtb.models.URDF.UR5()
    right_robot = rtb.models.URDF.UR5()
    left_robot.q = REAL_LEFT_ARM_STARTING_STATE
    right_robot.q = REAL_RIGHT_ARM_STARTING_STATE
    left_robot.base = sm.SE3(DISTANCE_BETWEEN_LEFT_RIGHT_ARMS, 0, 0)
    env_swift.add(left_robot)
    env_swift.add(right_robot)
else:
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
    if args.real_robot:
        create_new_episode_dirs_real(new_dir_for_augmented_images, org_dir)
    else:
        create_new_episode_dirs(new_dir_for_augmented_images, org_dir)

    w_l_filtered_cur_traj_json = {key: value for key, value in cur_traj_json.items() if 'w_l_diffusion_' in key}
    w_r_filtered_cur_traj_json = {key: value for key, value in cur_traj_json.items() if 'w_r_diffusion_' in key}

    # initialize some variables
    total_timesteps = len(low_dim_obs)
    new_dir_wrist_left_rgb_dir = os.path.join(new_dir_for_augmented_images, 'wrist_left_rgb')
    new_dir_wrist_right_rgb_dir = os.path.join(new_dir_for_augmented_images, 'wrist_right_rgb')

    bimanual_observations = []
    prev_original_image_index = -1
    ik_failures = 0
    saved_img_index = 0
    save_org_frames = True
    # for (w_l_aug_image_name, w_l_aug_image_data), (w_r_aug_image_name, w_r_aug_image_data) in zip(sorted(w_l_filtered_cur_traj_json.items()), sorted(w_r_filtered_cur_traj_json.items())):
    for w_l_aug_image_name, w_r_aug_image_name in zip(sorted(w_l_filtered_cur_traj_json.keys()), sorted(w_r_filtered_cur_traj_json.keys())):
        w_l_aug_image_data = w_l_filtered_cur_traj_json[w_l_aug_image_name]
        w_r_aug_image_data = w_r_filtered_cur_traj_json[w_r_aug_image_name]
        original_image_index = int(w_l_aug_image_data[3].split('.')[0].split('_')[-1])

        assert int(w_l_aug_image_data[3].split('.')[0].split('_')[-1]) == int(w_r_aug_image_data[3].split('.')[0].split('_')[-1])

        if original_image_index == 0:
            # save the first frame as is
            save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, 0, depth_npy=args.depth_npy)
            saved_img_index += 1
            prev_original_image_index = 0
            continue

        # copy (prev_original_image_index+1) state up to (and excluding) original_image_index state's bimanual observations and add them to the list
        # print(f'prev_original_image_index: {prev_original_image_index}, original_image_index: {original_image_index}') # for debugging
        for j in range(prev_original_image_index+1, original_image_index):
            # print('Original observation index:', j)
            local_bimanual_obs = copy.deepcopy(low_dim_obs[j])
            bimanual_observations.append(local_bimanual_obs)
            save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, j, depth_npy=args.depth_npy)
            saved_img_index += 1

        # NOTE: here we assume that only joint positions and end-effector pose are modified and different from the original data

        if args.real_robot:
            #################### left arm ####################
            # get the original left arm positions
            left_arm_joint_positions = low_dim_obs[original_image_index].left.joint_positions
            left_arm_left_wrist_cam_pose = low_dim_obs[original_image_index].perception_data['wrist_left_pose']
            # make sure the robot joints are in the same position as the original data
            left_robot.q = left_arm_joint_positions[:6]
            env_swift.step(ENV_SWIFT_DT)

            # get left-wrist augmented image's position and orientation (in left wrist's camera frame)
            left_wrist_aug_transform = np.eye(4)
            left_wrist_aug_transform[:3,:3] = w_l_aug_image_data[6]
            left_wrist_aug_transform[:3,3] = w_l_aug_image_data[5]
            left_wrist_aug_transform = from_blender_frame(left_wrist_aug_transform)

            # transform current (I_t) left-wrist cam frame to left-wrist augmented cam frame (I^{tilde}_t)
            left_arm_left_wrist_cam_aug_pose = np.dot(left_wrist_aug_transform, left_arm_left_wrist_cam_pose)

            # get the new left camera/end-effector position and orientation
            left_target = np.eye(4)
            left_rotation = R.from_matrix(left_arm_left_wrist_cam_aug_pose[:3,:3])
            left_quat = left_rotation.as_quat()
            ########!!!!!!!!!!!!!!! TODO: why does this work for lift ball and lift drawer tasks but not for push block task?
            left_target[:3,:3] = quaternion_to_rotation_matrix(np.array([left_quat[1], left_quat[2], left_quat[3], left_quat[0]])) # NOTE: this works for lift ball and lift drawer tasks
            # left_target[:3,:3] = quaternion_to_rotation_matrix(np.array([left_quat[0], left_quat[1], left_quat[2], left_quat[3]])) # NOTE: this works only for the push block task 
            left_target[:3, 3] = left_arm_left_wrist_cam_aug_pose[:3,3]
            left_sol = left_robot.ikine_LM(left_target, end="tool0", start="base", q0=left_robot.q, joint_limits=True)
            if left_sol.success:
                left_arm_new_joint_positions = left_sol.q
                left_robot.q = left_arm_new_joint_positions
                env_swift.step(ENV_SWIFT_DT)
                left_arm_new_joint_positions = np.concatenate((left_arm_new_joint_positions, [left_arm_joint_positions[-1]]))
            else:
                # IK failed for the left arm
                print('!!!!!!!!!! IK failed for left augmented image at index:', original_image_index)
                ik_failures += 1
                # save the left wrist original image to the new directory
                local_bimanual_obs = copy.deepcopy(low_dim_obs[original_image_index])
                bimanual_observations.append(local_bimanual_obs)

                save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, original_image_index, depth_npy=args.depth_npy)
                prev_original_image_index = original_image_index
                saved_img_index += 1

                # intuition is that the augmented state is most likely a bad one so we want to 
                # use the intermediate frames in the next frames.
                continue

            # get left arm's new gripper pose and matrix
            left_x, left_y, left_z = left_target[0, 0], left_target[0, 1], left_target[0, 2]
            left_arm_new_gripper_pose = [left_x, left_y, left_z, *left_rotation.as_quat()]
            left_arm_new_gripper_matrix = np.eye(4)
            left_arm_new_gripper_matrix[:3,:3] = left_rotation.as_matrix()
            left_arm_new_gripper_matrix[:3,3] = [left_x, left_y, left_z]

            # save the left wrist augmented image to the new directory
            dest_aug_left_wrist_img_path = os.path.join(new_dir_wrist_left_rgb_dir, f"rgb_{saved_img_index:04}.png")
            # dest_aug_left_wrist_img_path = os.path.join(new_dir_wrist_left_rgb_dir, f"rgb_{saved_img_index:04}_aug.png") # for debugging
            aug_left_wrist_img = Image.open(os.path.join(args.action_labels_dir, traj_folders[i], 'images', w_l_aug_image_name))
            # NOTE: we assume the original images are 128x128
            aug_left_wrist_img = aug_left_wrist_img.resize((128, 128))
            aug_left_wrist_img.save(dest_aug_left_wrist_img_path)

            #################### right arm ####################
            # get the original right arm positions
            right_arm_joint_positions = low_dim_obs[original_image_index].right.joint_positions
            right_arm_right_wrist_cam_pose = low_dim_obs[original_image_index].perception_data['wrist_right_pose']
            # make sure the robot joints are in the same position as the original data
            right_robot.q = right_arm_joint_positions[:6]
            env_swift.step(ENV_SWIFT_DT)

            # get right-wrist augmented image's position and orientation (in right wrist's camera frame)
            right_wrist_aug_transform = np.eye(4)
            right_wrist_aug_transform[:3,:3] = w_r_aug_image_data[6]
            right_wrist_aug_transform[:3,3] = w_r_aug_image_data[5]
            right_wrist_aug_transform = from_blender_frame(right_wrist_aug_transform)

            # transform current (I_t) right-wrist cam frame to right-wrist augmented cam frame (I^{tilde}_t)
            right_arm_right_wrist_cam_aug_pose = np.dot(right_wrist_aug_transform, right_arm_right_wrist_cam_pose)

            # get the new right end-effector position and orientation
            right_target = np.eye(4)
            right_rotation = R.from_matrix(right_arm_right_wrist_cam_aug_pose[:3,:3])
            right_quat = right_rotation.as_quat()
            right_target[:3,:3] = quaternion_to_rotation_matrix(np.array([right_quat[1], right_quat[2], right_quat[3], right_quat[0]]))
            right_target[:3, 3] = right_arm_right_wrist_cam_aug_pose[:3,3]
            right_sol = right_robot.ikine_LM(right_target, end="tool0", start="base", q0=right_robot.q, joint_limits=True)
            if right_sol.success:
                right_arm_new_joint_positions = right_sol.q
                right_robot.q = right_arm_new_joint_positions
                env_swift.step(ENV_SWIFT_DT)
                right_arm_new_joint_positions = np.concatenate((right_arm_new_joint_positions, [right_arm_joint_positions[-1]]))
            else:
                # IK failed for the right arm
                print('!!!!!!!!!! IK failed for right augmented image at index:', original_image_index)
                ik_failures += 1
                # save the right wrist original image to the new directory
                local_bimanual_obs = copy.deepcopy(low_dim_obs[original_image_index])
                bimanual_observations.append(local_bimanual_obs)
                
                save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, original_image_index, depth_npy=args.depth_npy)
                prev_original_image_index = original_image_index
                saved_img_index += 1
                continue

            # get right arm's new gripper pose and matrix
            right_x, right_y, right_z = right_target[0, 0], right_target[0, 1], right_target[0, 2]
            right_arm_new_gripper_pose = [right_x, right_y, right_z, *right_rotation.as_quat()]
            right_arm_new_gripper_matrix = np.eye(4)
            right_arm_new_gripper_matrix[:3,:3] = right_rotation.as_matrix()
            right_arm_new_gripper_matrix[:3,3] = [right_x, right_y, right_z]

            # save the right wrist augmented image to the new directory
            dest_aug_right_wrist_img_path = os.path.join(new_dir_wrist_right_rgb_dir, f"rgb_{saved_img_index:04}.png")
            # dest_aug_right_wrist_img_path = os.path.join(new_dir_wrist_right_rgb_dir, f"rgb_{saved_img_index:04}_aug.png") # for debugging
            aug_right_wrist_img = Image.open(os.path.join(args.action_labels_dir, traj_folders[i], 'images', w_r_aug_image_name))
            # NOTE: we assume the original images are 128x128
            aug_right_wrist_img = aug_right_wrist_img.resize((128, 128))
            aug_right_wrist_img.save(dest_aug_right_wrist_img_path)
            # print('augmented observation index:', original_image_index)
            saved_img_index += 1
        else:
            # not args.real_robot
            #################### left arm ####################
            # get the original left arm positions
            left_arm_joint_positions = low_dim_obs[original_image_index].left.joint_positions
            left_arm_left_wrist_cam_pose = low_dim_obs[original_image_index].perception_data['wrist_left_pose']
            # make sure the robot joints are in the same position as the original data
            robot.left_arm.set_joint_positions(left_arm_joint_positions, disable_dynamics=True)
            step_in_pyrep(pyrep)

            # get left-wrist augmented image's position and orientation (in left wrist's camera frame)
            left_wrist_aug_transform = np.eye(4)
            left_wrist_aug_transform[:3,:3] = w_l_aug_image_data[6]
            left_wrist_aug_transform[:3,3] = w_l_aug_image_data[5]
            left_wrist_aug_transform = from_blender_frame(left_wrist_aug_transform)

            # transform current (I_t) left-wrist cam frame to left-wrist augmented cam frame (I^{tilde}_t)
            left_arm_left_wrist_cam_aug_pose = np.dot(left_wrist_aug_transform, left_arm_left_wrist_cam_pose)
            # transform left-wrist cam frame to left end-effector frame
            t_left_wrist_cam_to_left_eff = np.dot(np.linalg.inv(low_dim_obs[original_image_index].perception_data['wrist_left_pose']), low_dim_obs[original_image_index].left.gripper_matrix)
            # transform left-wrist augmented cam frame (I^{tilde}_t) to its end-effector frame
            left_eff_frame = np.dot(left_arm_left_wrist_cam_aug_pose, t_left_wrist_cam_to_left_eff)
            # get the new left end-effector position and orientation
            left_x, left_y, left_z = left_eff_frame[:3,3]
            left_rotation_matrix_eff_frame = left_eff_frame[:3,:3]
            left_rotation = R.from_matrix(left_rotation_matrix_eff_frame)
            left_quaternion = left_rotation.as_quat()

            # use IK to figure out new joint positions
            try:
                left_arm_new_joint_positions = robot.left_arm.solve_ik_via_sampling([left_x, left_y, left_z], quaternion=left_quaternion)[0]
            except:
                # for debugging purposes
                # robot.left_arm.get_tip().get_quaternion()
                # robot.left_arm.solve_ik_via_sampling([left_x, left_y, left_z], quaternion=robot.left_arm.get_tip().get_quaternion())[0]
                # robot.left_arm.solve_ik_via_sampling(robot.left_arm.get_tip().get_position(), quaternion=left_quaternion)[0]
                # robot.left_arm.solve_ik_via_sampling([left_x, left_y, left_z], ignore_collisions=True, max_time_ms=100, quaternion=left_quaternion)[0]
                print('!!!!!!!!!! IK failed for left augmented image at index:', original_image_index)
                ik_failures += 1
                # save the left wrist original image to the new directory
                local_bimanual_obs = copy.deepcopy(low_dim_obs[original_image_index])
                bimanual_observations.append(local_bimanual_obs)

                save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, original_image_index, depth_npy=args.depth_npy)
                prev_original_image_index = original_image_index
                saved_img_index += 1

                # intuition is that the augmented state is most likely a bad one so we want to 
                # use the intermediate frames in the next frames.
                continue

            robot.left_arm.set_joint_positions(left_arm_new_joint_positions, disable_dynamics=True)
            # get left arm's new gripper pose and matrix
            left_arm_new_gripper_pose = [left_x, left_y, left_z, *left_quaternion]
            left_arm_new_gripper_matrix = np.eye(4)
            left_arm_new_gripper_matrix[:3,:3] = left_rotation_matrix_eff_frame
            left_arm_new_gripper_matrix[:3,3] = [left_x, left_y, left_z]

            # save the left wrist augmented image to the new directory
            dest_aug_left_wrist_img_path = os.path.join(new_dir_wrist_left_rgb_dir, f"rgb_{saved_img_index:04}.png")
            # dest_aug_left_wrist_img_path = os.path.join(new_dir_wrist_left_rgb_dir, f"rgb_{saved_img_index:04}_aug.png") # for debugging
            aug_left_wrist_img = Image.open(os.path.join(args.action_labels_dir, traj_folders[i], 'images', w_l_aug_image_name))
            # NOTE: we assume the original images are 128x128
            aug_left_wrist_img = aug_left_wrist_img.resize((128, 128))
            aug_left_wrist_img.save(dest_aug_left_wrist_img_path)

            #################### right arm ####################
            # get the original right arm positions
            right_arm_joint_positions = low_dim_obs[original_image_index].right.joint_positions
            right_arm_right_wrist_cam_pose = low_dim_obs[original_image_index].perception_data['wrist_right_pose']
            # make sure the robot joints are in the same position as the original data
            robot.right_arm.set_joint_positions(right_arm_joint_positions, disable_dynamics=True)
            step_in_pyrep(pyrep)

            # get right-wrist augmented image's position and orientation (in right wrist's camera frame)
            right_wrist_aug_transform = np.eye(4)
            right_wrist_aug_transform[:3,:3] = w_r_aug_image_data[6]
            right_wrist_aug_transform[:3,3] = w_r_aug_image_data[5]
            right_wrist_aug_transform = from_blender_frame(right_wrist_aug_transform)

            # transform current (I_t) right-wrist cam frame to right-wrist augmented cam frame (I^{tilde}_t)
            right_arm_right_wrist_cam_aug_pose = np.dot(right_wrist_aug_transform, right_arm_right_wrist_cam_pose)
            # transform right-wrist cam frame to right end-effector frame
            t_right_wrist_cam_to_right_eff = np.dot(np.linalg.inv(low_dim_obs[original_image_index].perception_data['wrist_right_pose']), low_dim_obs[original_image_index].right.gripper_matrix)
            # transform right-wrist augmented cam frame (I^{tilde}_t) to its end-effector frame
            right_eff_frame = np.dot(right_arm_right_wrist_cam_aug_pose, t_right_wrist_cam_to_right_eff)
            # get the new right end-effector position and orientation
            right_x, right_y, right_z = right_eff_frame[:3,3]
            right_rotation_matrix_eff_frame = right_eff_frame[:3,:3]
            right_rotation = R.from_matrix(right_rotation_matrix_eff_frame)
            right_quaternion = right_rotation.as_quat()

            # use IK to figure out new joint positions
            try:
                right_arm_new_joint_positions = robot.right_arm.solve_ik_via_sampling([right_x, right_y, right_z], quaternion=right_quaternion)[0]
            except:
                print('!!!!!!!!!! IK failed for right augmented image at index:', original_image_index)
                ik_failures += 1
                # save the right wrist original image to the new directory
                local_bimanual_obs = copy.deepcopy(low_dim_obs[original_image_index])
                bimanual_observations.append(local_bimanual_obs)
                
                save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, original_image_index, depth_npy=args.depth_npy)
                prev_original_image_index = original_image_index
                saved_img_index += 1
                continue

            robot.right_arm.set_joint_positions(right_arm_new_joint_positions, disable_dynamics=True)
            # get right arm's new gripper pose and matrix
            right_arm_new_gripper_pose = [right_x, right_y, right_z, *right_quaternion]
            right_arm_new_gripper_matrix = np.eye(4)
            right_arm_new_gripper_matrix[:3,:3] = right_rotation_matrix_eff_frame
            right_arm_new_gripper_matrix[:3,3] = [right_x, right_y, right_z]

            # save the right wrist augmented image to the new directory
            dest_aug_right_wrist_img_path = os.path.join(new_dir_wrist_right_rgb_dir, f"rgb_{saved_img_index:04}.png")
            # dest_aug_right_wrist_img_path = os.path.join(new_dir_wrist_right_rgb_dir, f"rgb_{saved_img_index:04}_aug.png") # for debugging
            aug_right_wrist_img = Image.open(os.path.join(args.action_labels_dir, traj_folders[i], 'images', w_r_aug_image_name))
            # NOTE: we assume the original images are 128x128
            aug_right_wrist_img = aug_right_wrist_img.resize((128, 128))
            aug_right_wrist_img.save(dest_aug_right_wrist_img_path)
            # print('augmented observation index:', original_image_index)
            saved_img_index += 1

        # We need to update right, left, perception_data, and misc. Ignore task_low_dim_state because it's not used. Ignore misc because nothing needs to be updated.
        # step 1: copy the original data
        bimanual_obs = copy.deepcopy(low_dim_obs[original_image_index])
        # step 2: update left and right objects' arm joint_positions, gripper_pose, gripper_matrix
        bimanual_obs.left.joint_positions = left_arm_new_joint_positions
        bimanual_obs.left.gripper_pose = left_arm_new_gripper_pose
        bimanual_obs.left.gripper_matrix = left_arm_new_gripper_matrix
        bimanual_obs.right.joint_positions = right_arm_new_joint_positions
        bimanual_obs.right.gripper_pose = right_arm_new_gripper_pose
        bimanual_obs.right.gripper_matrix = right_arm_new_gripper_matrix
        # step 3: update the perception_data, specifically "wrist_left_pose," which is the left wrist camera pose, and "wrist_right_pose"
        bimanual_obs.perception_data['wrist_left_pose'] = left_arm_left_wrist_cam_aug_pose
        bimanual_obs.perception_data['wrist_right_pose'] = right_arm_right_wrist_cam_aug_pose

        bimanual_observations.append(bimanual_obs)
        prev_original_image_index = original_image_index

    total_ik_failures[curr_folder] = ik_failures

    # add the remaining observations to the list
    for j in range(prev_original_image_index+1, total_timesteps):
        local_bimanual_obs = copy.deepcopy(low_dim_obs[j])
        bimanual_observations.append(local_bimanual_obs)
        save_org_left_right_wrist_images(org_dir, new_dir_for_augmented_images, saved_img_index, j, depth_npy=args.depth_npy) 
        saved_img_index += 1

    # NOTE: here, we assume the new demo has the same number of observations as the original demo
    new_demo = Demo(bimanual_observations, low_dim_obs.random_seed)
    # save new_demo to a pickle file
    with open(os.path.join(new_dir_for_augmented_images, 'low_dim_obs.pkl'), 'wb') as f:
        pickle.dump(new_demo, f)
    if args.real_robot:
        reset_bimanual_real_robot(env_swift, left_robot, right_robot)
    else:
        reset_bimanual(pyrep, robot, initial_robot_state, start_arm_joint_pos, starting_gripper_joint_pos)
    pbar.update(1)

if not args.real_robot:
    pyrep.stop()
    pyrep.shutdown()
pbar.close()
print('Finished generating augmented data!')
print('total_ik_failures: ', total_ik_failures)