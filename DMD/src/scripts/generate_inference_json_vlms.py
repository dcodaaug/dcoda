import os
import sys
import argparse
import pathlib
import json
import numpy as np
from scipy.spatial.transform import Rotation as scir

script_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(os.path.join(src_path, "datasets"))
sys.path.append('/'.join(src_path.split('/')[:-2]))

from pyrep import PyRep
from pyrep.robots.arms.dual_panda import PandaLeft
from pyrep.robots.arms.dual_panda import PandaRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperLeft
from rlbench.backend.robot import BimanualRobot
from rlbench.demo import Demo

from write_translations import get_colmap_labels
from slam_utils import read_pose, read_bimanual_pose, read_gripper_state_bimanual, read_gripper_matrices_bimanual
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import torch
import shutil
import cv2
from scipy.stats import zscore
from scipy.optimize import dual_annealing
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import time
import math
from skimage.metrics import structural_similarity as ssim

MINIMUM_CONTACT_POINTS = 1000
# MINIMUM_CONTACT_POINTS = 30 # used this for depth images in npy format
MAXIMUM_GRIPPER_DEPTH = 0.15
MINIMUM_OBJ_COST = 0.0
DEPTH_SCALE = 2**24 - 1
DEFAULT_RGB_SCALE_FACTOR = 256000.0
DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}
DISTANCE_BETWEEN_LEFT_RIGHT_ARMS = 1.22 # meters
REAL_LEFT_ARM_STARTING_STATE = [1.3629151365049208, -1.1219412063110294, -2.036740639010723, -1.632770313019793, 1.7637373624131287, 2.771111558000195]
REAL_RIGHT_ARM_STARTING_STATE = [-1.4106484960522758, -1.9944810024925443, 2.054192150410259, -1.5718085742732786, -1.6140418455715997, 0.061382133371988376]
ENV_SWIFT_DT = 0.05
REAL_TOOL_TIP_MINIMUM_Z = 0.018
REAL_MINIMUM_DIST_BETWEEN_EFFS = 0.05 # meters
OUTPUT_DIR = '../data/traj_plots'
ORG_LEFT_GRIPPER_IMG = None
ORG_RIGHT_GRIPPER_IMG = None
SSIM_THRESHOLD = 0.995

current_time = time.time()
def image_to_float_array(image, scale_factor=None):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

    Returns:
    A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array

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

def create_folder_if_not_exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    # Remove all contents in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            shutil.rmtree(file_path) if os.path.isdir(file_path) else os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def get_gripper_masks(predictor, images_folder, gripper_mask, frame_indices, which_arm="left"):
    """
    Use SAM2 to generate gripper masks for a trajectory of images.
    """
    inference_state = predictor.init_state(video_path=images_folder)
    predictor.reset_state(inference_state)
    _, out_obj_ids, video_res_masks = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=gripper_mask,
    )
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    non_dilated_video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        local_dict = {}
        non_dilated_local_dict = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            final_mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
            # make sure the top half of the mask is all zeros
            half_height = int(final_mask.shape[1]/2)
            final_mask[0, :half_height, :] = 0
            # Apply dilation
            processed_final_mask = cv2.dilate(final_mask, np.ones((23, 15), np.uint8), iterations=1)
            # Apply morphological closing
            processed_final_mask = cv2.morphologyEx(processed_final_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            non_dilated_local_dict[out_obj_id] = final_mask
            local_dict[out_obj_id] = processed_final_mask
        video_segments[out_frame_idx] = local_dict
        non_dilated_video_segments[out_frame_idx] = non_dilated_local_dict

    # debugging: visualize gripper segmentation results
    # import matplotlib.pyplot as plt
    # def show_mask(mask, ax, obj_id=None, random_color=False):
    #     if random_color:
    #         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #     else:
    #         cmap = plt.get_cmap("tab10")
    #         cmap_idx = 0 if obj_id is None else obj_id
    #         color = np.array([*cmap(cmap_idx)[:3], 0.6])
    #     h, w = mask.shape[-2:]
    #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #     ax.imshow(mask_image)
    # plt.close("all")
    # for out_frame_idx in range(0, len(frame_indices)):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(f"{images_folder}/{frame_indices[out_frame_idx]}.jpg"))
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    #     plt.savefig(f"../data/output/dilated_{which_arm}_{out_frame_idx}.png", format="png")
    # for out_frame_idx in range(0, len(frame_indices)):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(f"{images_folder}/{frame_indices[out_frame_idx]}.jpg"))
    #     for out_obj_id, out_mask in non_dilated_video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    #     plt.savefig(f"../data/output/{which_arm}_{out_frame_idx}.png", format="png")

    return video_segments, non_dilated_video_segments

def _check_nearby_objects_within_proximity(real_robot, depth_image, gripper_depth, gripper_mask, non_dilated_gripper_mask, idx, frame_idx, left_wrist_img, right_wrist_img, ep_num=None):
    if real_robot:
        global ORG_LEFT_GRIPPER_IMG
        global ORG_RIGHT_GRIPPER_IMG
        left_gripper_img = left_wrist_img * np.stack([non_dilated_gripper_mask]*3, axis=-1)
        right_gripper_img = right_wrist_img * np.stack([non_dilated_gripper_mask]*3, axis=-1)
        if idx == 0:
            ORG_LEFT_GRIPPER_IMG = left_gripper_img
            ORG_RIGHT_GRIPPER_IMG = right_gripper_img
            # assume there is no contact at the first frame
            return False
        else:
            left_score, _ = ssim(ORG_LEFT_GRIPPER_IMG, left_gripper_img, full=True, channel_axis=2)
            right_score, _ = ssim(ORG_RIGHT_GRIPPER_IMG, right_gripper_img, full=True, channel_axis=2)
            # print(f'idx: {idx}, frame_idx: {frame_idx}')
            # print(f'left score: {left_score}, right score: {right_score}')
            # cv2.imwrite("../data/left_gripper_img.png", left_gripper_img)
            # cv2.imwrite("../data/right_gripper_img.png", right_gripper_img)
            if left_score < SSIM_THRESHOLD or right_score < SSIM_THRESHOLD:
                # something is blocking the gripper, so we assume there is contact
                print(f'Idx contact: {idx} frame_idx: {frame_idx}, Grippers are blocked, which means there must be contact')
                return True
            else:
                print(f'Idx contact: {idx} frame_idx: {frame_idx}, no contact')
                return False

    ########### Filter out the gripper depth values that are not within the gripper mask (outliers) ###########
    non_dilated_gripper_depth = depth_image * non_dilated_gripper_mask
    # Compute z-scores
    z_scores = zscore(non_dilated_gripper_depth)
    # Define a threshold for outliers (3 is a common value)
    threshold = 2.2
    # Filter out outliers
    non_dilated_gripper_depth = non_dilated_gripper_depth[np.abs(z_scores) < threshold]

    ########### Filter out gripper depth values that are greater than the maximum gripper depth ###########
    non_dilated_gripper_depth = non_dilated_gripper_depth[non_dilated_gripper_depth < MAXIMUM_GRIPPER_DEPTH]
    ########### Find the largest gripper depth value excluding outliers ###########
    if non_dilated_gripper_depth.shape[0] == 0:
        print(f'Idx contact: {idx} frame_idx: {frame_idx}, Grippers are not visible, which means there must be contact')
        return True
    else:
        max_gripper_depth = np.max(non_dilated_gripper_depth)

    ########### Find the contour of the gripper mask and fill in the missing parts to deal with incomplete segmentation masks ###########
    contours, _ = cv2.findContours(gripper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gripper_mask_filled = np.zeros_like(gripper_mask)
    # Iterate over contours to find the bounding box and complete the rectangle
    for contour in contours:
        # Fill the current contour
        cv2.drawContours(gripper_mask_filled, [contour], -1, color=1, thickness=cv2.FILLED)
        # Get the bounding rectangle of the current contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a rectangle to fill the missing parts
        cv2.rectangle(gripper_mask_filled, (x, y), (x + w, y + h), color=1, thickness=cv2.FILLED)
    cv2.imwrite("../data/gripper_mask_filled.png", cv2.applyColorMap(cv2.normalize(gripper_mask_filled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)) # debugging: visualize filled gripper mask

    ###########  Use dilated gripper mask to extract depth image without gripper. Why dilated? Because the gripper mask may not covert the entire gripper, so we need to dialate it to make sure we are not missing any part of the gripper. ########### 
    depth_without_gripper = np.where(gripper_mask_filled == 1, np.inf, depth_image)
    closest_point_to_gripper = np.min(depth_without_gripper)

    ########### Check if the closest point to the gripper is touching the gripper ###########
    contact = False
    if closest_point_to_gripper <= max_gripper_depth:
        contact = True
    
    # debugging: visualize depth images
    # cv2.imwrite("../data/depth_image_rgb.png", cv2.applyColorMap(cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
    # non_dilated_gripper_depth_img = depth_image * non_dilated_gripper_mask; cv2.imwrite("../data/non_dilated_gripper_depth_img.png", cv2.applyColorMap(cv2.normalize(non_dilated_gripper_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
    # cv2.imwrite("../data/gripper_depth_rgb.png", cv2.applyColorMap(cv2.normalize(gripper_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
    # depth_without_gripper = np.where(gripper_mask_filled == 1, 2, depth_image); cv2.imwrite("../data/depth_rgb_without_gripper.png", cv2.applyColorMap(cv2.normalize(depth_without_gripper, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
    # depth_without_gripper = np.where(gripper_mask_filled == 1, np.inf, depth_image); contact = (depth_without_gripper <= max_gripper_depth).astype(int); cv2.imwrite("../data/contact_map.png", cv2.applyColorMap(cv2.normalize(contact, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))

    ########### Handles the case where there are small holes or partially excluded gripper in the gripper segmentation mask ###########
    contact = (depth_without_gripper <= max_gripper_depth).astype(int)
    num_contact_points = np.sum(contact)
    if num_contact_points > MINIMUM_CONTACT_POINTS:
        contact = True
        print(f'Idx contact: {idx}, frame_idx {frame_idx}, num_contact_points: {num_contact_points}') # for debugging
        if idx <= 12:
            print("WARNING: Double check results in _check_nearby_objects_within_proximity since idx shouldn't be so small.")
    else:
        contact = False
        print(f'Idx: {idx}, frame_idx: {frame_idx}') # for debugging

    return contact

def determine_if_contact_occurred(real_robot, left_wrist_depth_img_path, right_wrist_depth_img_path, depth_info, idx, frame_idx, left_gripper_video_segments, right_gripper_video_segments, non_dilated_left_gripper_video_segments, non_dilated_right_gripper_video_segments, load_depth_npy, left_full_img_path, right_full_img_path, ep_num=None):
    left_wrist_img = Image.open(left_wrist_depth_img_path)
    right_wrist_img = Image.open(right_wrist_depth_img_path)
    if load_depth_npy:
        left_wrist_depth_img = np.load(left_wrist_depth_img_path, allow_pickle=True)
        right_wrist_depth_img = np.load(right_wrist_depth_img_path, allow_pickle=True)
    else:
        left_wrist_depth_img = image_to_float_array(Image.open(left_wrist_depth_img_path), DEPTH_SCALE)
        right_wrist_depth_img = image_to_float_array(Image.open(right_wrist_depth_img_path), DEPTH_SCALE)

    wrist_left_depth_info = depth_info[left_wrist_depth_img_path.split('/')[-1]]
    wrist_right_depth_info = depth_info[right_wrist_depth_img_path.split('/')[-1]]

    if real_robot:
        wrist_left_depth_image_m = left_wrist_depth_img / 1000.0 # convert to meters
        wrist_right_depth_image_m = right_wrist_depth_img / 1000.0 # convert to meters
    else:
        # convert depth images to meters using equations found in RLBench/rlbench/utils.py
        wrist_left_depth_image_m = wrist_left_depth_info['wrist_left_camera_near'] + left_wrist_depth_img * (wrist_left_depth_info['wrist_left_camera_far'] - wrist_left_depth_info['wrist_left_camera_near'])
        wrist_right_depth_image_m = wrist_right_depth_info['wrist_right_camera_near'] + right_wrist_depth_img * (wrist_right_depth_info['wrist_right_camera_far'] - wrist_right_depth_info['wrist_right_camera_near'])

    # current gripper masks
    left_gripper_mask = left_gripper_video_segments[idx][1][0]
    right_gripper_mask = right_gripper_video_segments[idx][1][0]
    non_dilated_left_gripper_mask = non_dilated_left_gripper_video_segments[idx][1][0]
    non_dilated_right_gripper_mask = non_dilated_right_gripper_video_segments[idx][1][0]

    # current gripper depth images
    left_gripper_depth_img = wrist_left_depth_image_m * left_gripper_mask
    right_gripper_depth_img = wrist_right_depth_image_m * right_gripper_mask

    left_gripper_contact = _check_nearby_objects_within_proximity(real_robot, wrist_left_depth_image_m, left_gripper_depth_img, left_gripper_mask, non_dilated_left_gripper_mask, idx, frame_idx, left_wrist_img, right_wrist_img, ep_num=ep_num)
    right_gripper_contact = _check_nearby_objects_within_proximity(real_robot, wrist_right_depth_image_m, right_gripper_depth_img, right_gripper_mask, non_dilated_right_gripper_mask, idx, frame_idx, left_wrist_img, right_wrist_img, ep_num=ep_num)

    return left_gripper_contact or right_gripper_contact

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

def normalize(x, bounds):
    """
    Function to normalize variables from original bounds to [-1, 1]
    """
    return [2 * (xi - b[0]) / (b[1] - b[0]) - 1 for xi, b in zip(x, bounds)]

def unnormalize(x, bounds):
    """
    Function to unnormalize variables from [-1, 1] to original bounds
    """
    return [(xi + 1) * (b[1] - b[0]) / 2 + b[0] for xi, b in zip(x, bounds)]

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

def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two 3D points.

    Parameters:
        point1 (tuple or list): A 3D point (x, y, z)
        point2 (tuple or list): A 3D point (x, y, z)

    Returns:
        float: The Euclidean distance between point1 and point2.
    """
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    dz = point1[2] - point2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def objective_function(x, robot, og_bounds, left_wrist_cam_pose, right_wrist_cam_pose, frame_idx, magnitude_l, 
                        x_euler_range, y_euler_range, z_euler_range, left_dct_gripper_matrix,
                            right_dct_gripper_matrix, debug_results=False):
    cost = MINIMUM_OBJ_COST

    # unnormalize the variables
    variables = unnormalize(x, og_bounds)
    trans = np.array(variables)

    # penalize the cost if the translation is too small
    if np.any(np.logical_and(trans > -magnitude_l, trans < magnitude_l)):
        cost += 10

    #################### left arm ####################
    # prepare the transformation matrix
    left_wrist_aug_transform = np.eye(4)
    left_wrist_aug_transform[:3,3] = trans
    left_wrist_aug_transform = from_blender_frame(left_wrist_aug_transform)

    # transform current (I_t) left-wrist cam frame to left-wrist augmented cam frame (I^{tilde}_t)
    left_arm_left_wrist_cam_aug_pose = np.dot(left_wrist_aug_transform, left_wrist_cam_pose)
    # transform left-wrist cam frame to left end-effector frame
    t_left_wrist_cam_to_left_eff = np.dot(np.linalg.inv(left_wrist_cam_pose), left_dct_gripper_matrix)
    # transform left-wrist augmented cam frame (I^{tilde}_t) to its end-effector frame
    left_eff_frame = np.dot(left_arm_left_wrist_cam_aug_pose, t_left_wrist_cam_to_left_eff)
    # get the new left end-effector position and orientation
    left_x, left_y, left_z = left_eff_frame[:3,3]
    left_rotation_matrix_eff_frame = left_eff_frame[:3,:3]
    left_rotation = R.from_matrix(left_rotation_matrix_eff_frame)
    left_quaternion = left_rotation.as_quat()
    
    try:
        left_arm_new_joint_positions = robot.left_arm.solve_ik_via_sampling([left_x, left_y, left_z], quaternion=left_quaternion)[0]
        # robot.left_arm.set_joint_positions(left_arm_new_joint_positions, disable_dynamics=True)
    except:
        # IK failed for the left arm
        cost += 10
    
    #################### right arm ####################
    # prepare the transformation matrix
    right_wrist_aug_transform = np.eye(4)
    right_wrist_aug_transform[:3,3] = trans
    right_wrist_aug_transform = from_blender_frame(right_wrist_aug_transform)

    # transform current (I_t) right-wrist cam frame to right-wrist augmented cam frame (I^{tilde}_t)
    right_arm_right_wrist_cam_aug_pose = np.dot(right_wrist_aug_transform, right_wrist_cam_pose)
    # transform right-wrist cam frame to right end-effector frame
    t_right_wrist_cam_to_right_eff = np.dot(np.linalg.inv(right_wrist_cam_pose), right_dct_gripper_matrix)
    # transform right-wrist augmented cam frame (I^{tilde}_t) to its end-effector frame
    right_eff_frame = np.dot(right_arm_right_wrist_cam_aug_pose, t_right_wrist_cam_to_right_eff)
    # get the new right end-effector position and orientation
    right_x, right_y, right_z = right_eff_frame[:3,3]
    right_rotation_matrix_eff_frame = right_eff_frame[:3,:3]
    right_rotation = R.from_matrix(right_rotation_matrix_eff_frame)
    right_quaternion = right_rotation.as_quat()
    
    try:
        right_arm_new_joint_positions = robot.right_arm.solve_ik_via_sampling([right_x, right_y, right_z], quaternion=right_quaternion)[0]
        # robot.right_arm.set_joint_positions(right_arm_new_joint_positions, disable_dynamics=True)
    except:
        # IK failed for the right arm
        cost += 10

    if debug_results:
        debug_results_dict = {
            'left_arm_new_joint_positions': left_arm_new_joint_positions,
            'right_arm_new_joint_positions': right_arm_new_joint_positions,
        }
        return cost, debug_results_dict

    return cost

def objective_function_no_contacts(x, robot, og_bounds, left_wrist_cam_pose, right_wrist_cam_pose, frame_idx, magnitude_l, 
                        x_euler_range, y_euler_range, z_euler_range, left_dct_gripper_matrix,
                            right_dct_gripper_matrix, debug_results=False):
    cost = MINIMUM_OBJ_COST

    # unnormalize the variables
    variables = unnormalize(x, og_bounds)
    trans_left = np.array(variables[:3])
    rot_euler_left = np.array(variables[3:6])
    rot_euler_right = np.array(variables[6:9])
    trans_right = np.array(variables[9:])

    # penalize the cost if the translation is too small
    if np.any(np.logical_and(trans_left > -magnitude_l, trans_left < magnitude_l)):
        cost += 10
    if np.any(np.logical_and(trans_right > -magnitude_l, trans_right < magnitude_l)):
        cost += 10

    #################### left arm ####################
    # prepare the transformation matrix
    left_wrist_aug_transform = np.eye(4)
    left_wrist_aug_transform[:3,:3] = scir.from_euler('xyz', rot_euler_left, degrees=True).as_matrix()
    left_wrist_aug_transform[:3,3] = trans_left
    left_wrist_aug_transform = from_blender_frame(left_wrist_aug_transform)

    # transform current (I_t) left-wrist cam frame to left-wrist augmented cam frame (I^{tilde}_t)
    left_arm_left_wrist_cam_aug_pose = np.dot(left_wrist_aug_transform, left_wrist_cam_pose)
    # transform left-wrist cam frame to left end-effector frame
    t_left_wrist_cam_to_left_eff = np.dot(np.linalg.inv(left_wrist_cam_pose), left_dct_gripper_matrix)
    # transform left-wrist augmented cam frame (I^{tilde}_t) to its end-effector frame
    left_eff_frame = np.dot(left_arm_left_wrist_cam_aug_pose, t_left_wrist_cam_to_left_eff)
    # get the new left end-effector position and orientation
    left_x, left_y, left_z = left_eff_frame[:3,3]
    left_rotation_matrix_eff_frame = left_eff_frame[:3,:3]
    left_rotation = R.from_matrix(left_rotation_matrix_eff_frame)
    left_quaternion = left_rotation.as_quat()
    
    try:
        left_arm_new_joint_positions = robot.left_arm.solve_ik_via_sampling([left_x, left_y, left_z], quaternion=left_quaternion)[0]
    except:
        # IK failed for the left arm
        cost += 10
    
    #################### right arm ####################
    # prepare the transformation matrix
    right_wrist_aug_transform = np.eye(4)
    right_wrist_aug_transform[:3,:3] = scir.from_euler('xyz', rot_euler_right, degrees=True).as_matrix()
    right_wrist_aug_transform[:3,3] = trans_right
    right_wrist_aug_transform = from_blender_frame(right_wrist_aug_transform)

    # transform current (I_t) right-wrist cam frame to right-wrist augmented cam frame (I^{tilde}_t)
    right_arm_right_wrist_cam_aug_pose = np.dot(right_wrist_aug_transform, right_wrist_cam_pose)
    # transform right-wrist cam frame to right end-effector frame
    t_right_wrist_cam_to_right_eff = np.dot(np.linalg.inv(right_wrist_cam_pose), right_dct_gripper_matrix)
    # transform right-wrist augmented cam frame (I^{tilde}_t) to its end-effector frame
    right_eff_frame = np.dot(right_arm_right_wrist_cam_aug_pose, t_right_wrist_cam_to_right_eff)
    # get the new right end-effector position and orientation
    right_x, right_y, right_z = right_eff_frame[:3,3]
    right_rotation_matrix_eff_frame = right_eff_frame[:3,:3]
    right_rotation = R.from_matrix(right_rotation_matrix_eff_frame)
    right_quaternion = right_rotation.as_quat()
    
    try:
        right_arm_new_joint_positions = robot.right_arm.solve_ik_via_sampling([right_x, right_y, right_z], quaternion=right_quaternion)[0]
    except:
        # IK failed for the right arm
        cost += 10

    if debug_results:
        debug_results_dict = {
            'left_arm_new_joint_positions': left_arm_new_joint_positions,
            'right_arm_new_joint_positions': right_arm_new_joint_positions,
        }
        return cost, debug_results_dict

    return cost

def objective_function_real_robot(x, env_swift, left_robot, right_robot, og_bounds, left_wrist_cam_pose, right_wrist_cam_pose, frame_idx, magnitude_l, 
                        x_euler_range, y_euler_range, z_euler_range, left_dct_gripper_matrix,
                            right_dct_gripper_matrix, debug_results=False):
    cost = MINIMUM_OBJ_COST

    # unnormalize the variables
    variables = unnormalize(x, og_bounds)
    trans = np.array(variables)

    # penalize the cost if the translation is too small
    if np.any(np.logical_and(trans > -magnitude_l, trans < magnitude_l)):
        cost += 10

    reset_bimanual_real_robot(env_swift, left_robot, right_robot)

    #################### left arm ####################
    # prepare the transformation matrix
    left_wrist_aug_transform = np.eye(4)
    left_wrist_aug_transform[:3,3] = trans
    left_wrist_aug_transform = from_blender_frame(left_wrist_aug_transform)

    # transform current (I_t) left-wrist cam frame to left-wrist augmented cam frame (I^{tilde}_t)
    left_arm_left_wrist_cam_aug_pose = np.dot(left_wrist_aug_transform, left_wrist_cam_pose)
    # get the new left camera/end-effector position and orientation
    left_target = np.eye(4)
    left_rotation = R.from_matrix(left_arm_left_wrist_cam_aug_pose[:3,:3])
    left_quat = left_rotation.as_quat()
    # left_target[:3,:3] = quaternion_to_rotation_matrix(np.array([left_quat[1], left_quat[2], left_quat[3], left_quat[0]])) # NOTE: this is actually wrong for the left arm
    left_target[:3,:3] = quaternion_to_rotation_matrix(np.array([left_quat[0], left_quat[1], left_quat[2], left_quat[3]]))
    left_target[:3, 3] = left_arm_left_wrist_cam_aug_pose[:3,3]
    left_sol = left_robot.ikine_LM(left_target, end="tool0", start="base", q0=left_robot.q, joint_limits=True)
    if left_sol.success:
        left_arm_new_joint_positions = left_sol.q
        left_robot.q = left_arm_new_joint_positions
        env_swift.step(ENV_SWIFT_DT)
    else:
        # IK failed for the left arm
        # print('IK failed for the left arm')
        cost += 10

    #################### right arm ####################
    # prepare the transformation matrix
    right_wrist_aug_transform = np.eye(4)
    right_wrist_aug_transform[:3,3] = trans
    right_wrist_aug_transform = from_blender_frame(right_wrist_aug_transform)

    # transform current (I_t) right-wrist cam frame to right-wrist augmented cam frame (I^{tilde}_t)
    right_arm_right_wrist_cam_aug_pose = np.dot(right_wrist_aug_transform, right_wrist_cam_pose)
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
    else:
        # IK failed for the right arm
        # print('IK failed for the right arm')
        cost += 10
    
    if left_target[2, 3] <= REAL_TOOL_TIP_MINIMUM_Z or right_target[2, 3] <= REAL_TOOL_TIP_MINIMUM_Z:
        # penalize the cost if the end-effector is going to hit the table
        # print(f'penalize the cost if the end-effector is going to hit the table: left z {left_target[2, 3]}, org left z {left_wrist_cam_pose[2,3]}, right z {right_target[2, 3]}, org right z {right_wrist_cam_pose[2,3]}')
        cost += 10

    dist_between_effs = euclidean_distance(left_target[:3, 3], right_target[:3, 3])
    if dist_between_effs <= REAL_MINIMUM_DIST_BETWEEN_EFFS:
        # penalize the cost if the end-effectors are too close to each other
        # print('penalize the cost if the end-effectors are too close to each other')
        cost += 10

    # breakpoint()
    if debug_results:
        debug_results_dict = {
            'left_arm_new_joint_positions': left_arm_new_joint_positions,
            'right_arm_new_joint_positions': right_arm_new_joint_positions,
        }
        return cost, debug_results_dict

    return cost

def early_termination_callback(x, f, context):
    """
    Callback to stop the optimization when the target cost is reached.
    Parameters:
        x: Current parameter values
        f: Current objective function value
        context: Unused, but included as part of the signature
    """
    elapsed_time = time.time() - current_time  # Calculate elapsed time
    if f <= MINIMUM_OBJ_COST:
        return True  # Return True to stop the optimization
    elif elapsed_time >= 60:  # Check if 60 seconds have passed
        print('No optimal solution found within 60 seconds')
        return True
    return False

def add_line_segment_for_each_basis(i, traces, point, R, prefix=''):
    colors = ['red', 'green', 'blue']  # x, y, z axes
    for j in range(3):  # Iterate through x, y, z axes
        # Compute start and end points of the arrow
        start = point
        end = point + (R[:, j] / 30)  # Extend the basis vector
        
        # Add a line segment for each basis vector
        traces.append(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+markers',
            line=dict(color=colors[j], width=5),
            marker=dict(size=0.5),
            name=f"{prefix} {i} Axis {['X', 'Y', 'Z'][j]}",
        ))

def visualize_robot_trajectory(left_wrist_traj, left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj, contacts_list, ep_num):
    """
    Visualizes the trajectories and poses of robot's wrist cameras in 3D space.

    Parameters:
    left_wrist_traj (list of tuples): A list of tuples, where each tuple contains:
        - (x, y, z): The position of the wrist camera.
        - R: A 3x3 rotation matrix representing the orientation of the wrist camera.

    left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj have the same structure as left_wrist_traj.
    """
    index = 0
    traces = []
    l_positions = []
    r_positions = []
    l_a_no_contacts_positions = []
    r_a_no_contacts_positions = []
    l_a_contacts_positions = []
    r_a_contacts_positions = []
    for l_pose, l_a_pose, r_pose, r_a_pose, contact in zip(left_wrist_traj, left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj, contacts_list):
        l_position, l_R = l_pose
        l_a_position, l_a_R = l_a_pose
        r_position, r_R = r_pose
        r_a_position, r_a_R = r_a_pose

        l_positions.append(l_position)
        r_positions.append(r_position)

        add_line_segment_for_each_basis(index, traces, l_position, l_R, prefix='Org L')
        add_line_segment_for_each_basis(index, traces, r_position, r_R, prefix='Org R')

        if contact:
            l_a_contacts_positions.append(l_a_position)
            r_a_contacts_positions.append(r_a_position)
        else:
            l_a_no_contacts_positions.append(l_a_position)
            r_a_no_contacts_positions.append(r_a_position)
        add_line_segment_for_each_basis(index, traces, l_a_position, l_a_R, prefix='Aug L')
        add_line_segment_for_each_basis(index, traces, r_a_position, r_a_R, prefix='Aug R')
        index += 1

    l_positions = np.array(l_positions)
    r_positions = np.array(r_positions)
    l_a_no_contacts_positions = np.array(l_a_no_contacts_positions)
    r_a_no_contacts_positions = np.array(r_a_no_contacts_positions)
    l_a_contacts_positions = np.array(l_a_contacts_positions)
    r_a_contacts_positions = np.array(r_a_contacts_positions)
    traces.append(
        go.Scatter3d(
        x=l_positions[:,0],
        y=l_positions[:,1],
        z=l_positions[:,2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Left Wrist Original Trajectory',
    ))

    traces.append(
        go.Scatter3d(
        x=r_positions[:,0],
        y=r_positions[:,1],
        z=r_positions[:,2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Right Wrist Original Trajectory',
    ))

    if len(l_a_no_contacts_positions) > 0:
        traces.append(
            go.Scatter3d(
            x=l_a_no_contacts_positions[:,0],
            y=l_a_no_contacts_positions[:,1],
            z=l_a_no_contacts_positions[:,2],
            mode='markers',
            marker=dict(size=5, color='yellow'),
            name='Left Wrist Augmented Trajectory (No Contact)',
        ))

    if len(r_a_no_contacts_positions) > 0:
        traces.append(
            go.Scatter3d(
            x=r_a_no_contacts_positions[:,0],
            y=r_a_no_contacts_positions[:,1],
            z=r_a_no_contacts_positions[:,2],
            mode='markers',
            marker=dict(size=5, color='#638C6D'), # green
            name='Right Wrist Augmented Trajectory (No Contact)',
        ))

    if len(l_a_contacts_positions) > 0:
        traces.append(
            go.Scatter3d(
            x=l_a_contacts_positions[:,0],
            y=l_a_contacts_positions[:,1],
            z=l_a_contacts_positions[:,2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Left Wrist Augmented Trajectory (Contact Points)',
        ))

    if len(r_a_contacts_positions) > 0:
        traces.append(
            go.Scatter3d(
            x=r_a_contacts_positions[:,0],
            y=r_a_contacts_positions[:,1],
            z=r_a_contacts_positions[:,2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Right Wrist Augmented Trajectory (Contact Points)',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
        ),
        title='3D Visualization of Robot Trajectories'
    )
    save_path = f'{OUTPUT_DIR}/ep_{ep_num}_trajectory_plot.html'
    fig.write_html(save_path)
    print(f"Plot saved to {save_path}")

def main(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('Random seed:', args.seed)

    if args.suffix is not None:
        output_root = args.output_root + f"_{args.suffix}"
    else:
        output_root = args.output_root
    # create a folder to store all the generated images
    pathlib.Path(output_root+"/images").mkdir(parents=True, exist_ok=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.save_print_to_file:
        print_output_filepath = f'{OUTPUT_DIR}/print_output.txt'
        print_output_file = open(print_output_filepath, 'w')
        # Redirect standard output to the file
        sys.stdout = print_output_file
    
    data = {}
    for pf, focal_length, img_h in zip(args.data_folders, args.focal_lengths, args.image_heights):
        for folder_name in os.listdir(pf):
            folder_path = os.path.join(pf, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if args.sfm_method == "colmap":
                if os.path.exists(os.path.join(folder_path, "FAIL")):
                    continue
                dct = get_colmap_labels(folder_path, focal_length=focal_length, data_type="apple")
            elif args.sfm_method == "grabber_orbslam":
                json_path = os.path.join(folder_path, "raw_labels.json")
                if not os.path.exists(json_path):
                    print(f"Skipping {folder_path}, no raw_labels.json")
                    continue
                dct = read_pose(json_path)
            elif args.sfm_method == "orbslam_bimanual":
                json_path = os.path.join(folder_path, "raw_labels_bimanual.json")
                if not os.path.exists(json_path):
                    print(f"Skipping {folder_path}, no raw_labels_bimanual.json")
                    continue
                dct_left, dct_right = read_bimanual_pose(json_path)

                dct_gripper = None
                json_path = os.path.join(folder_path, "gripper_state_bimanual.json")
                if not os.path.exists(json_path):
                    print(f"Skipping {folder_path}, no gripper_state_bimanual.json")
                else:
                    dct_gripper = read_gripper_state_bimanual(json_path)

                dct_gripper_matrices = None
                json_path = os.path.join(folder_path, "gripper_matrices.json")
                if not os.path.exists(json_path):
                    print(f"Skipping {folder_path}, no gripper_matrices.json")
                else:
                    left_dct_gripper_matrices, right_dct_gripper_matrices = read_gripper_matrices_bimanual(json_path)

            if args.sfm_method == "orbslam_bimanual":
                if args.real_wl_focal_length is not None:
                     dct_left["focal_y"] = float(args.real_wl_focal_length/img_h)
                else:
                    dct_left["focal_y"] = float(focal_length/img_h)
                if args.real_wr_focal_length is not None:
                    dct_right["focal_y"] = float(args.real_wr_focal_length/img_h)
                else:
                    dct_right["focal_y"] = float(focal_length/img_h)
                data[folder_path] = {'dct_left': dct_left, 'dct_right': dct_right, 'dct_gripper': dct_gripper, 'dct_left_gripper_matrices': left_dct_gripper_matrices, 'dct_right_gripper_matrices': right_dct_gripper_matrices}
            else:
                dct["focal_y"] = float(focal_length/img_h)
                data[folder_path] = dct
            
            # read depth info
            json_path = os.path.join(folder_path, "depth_info.json")
            depth_info_json = json.load(open(json_path, "r"))
            data[folder_path]["depth_info"] = depth_info_json
    
    pre_conversion = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # Here we assume all of the tasks use the same range for x, y, z rotations and magnitudes
    x_euler_range = [-0.5,0.5]
    y_euler_range = [-0.5,0.5]
    z_euler_range = [-0.5,0.5]
    magnitude_l, magnitude_u = 0.01,0.02
    
    # initialize SAM2
    sam2_checkpoint = "../../data/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cuda"))

    # load gripper segmentation masks
    if args.real_robot:
        wl_gripper_mask = "../data/wrist_left_rgb_0000_real_mask.png"
        wr_gripper_mask = "../data/wrist_right_rgb_0000_real_mask.png"
    else:
        wl_gripper_mask = "../data/wrist_left_00000_mask.png"
        wr_gripper_mask = "../data/wrist_right_00000_mask.png"
    left_wrist_gripper_mask = Image.open(wl_gripper_mask).convert("L")
    left_wrist_gripper_mask = np.array(left_wrist_gripper_mask)
    left_wrist_gripper_mask = (left_wrist_gripper_mask > 125).astype(int)
    right_wrist_gripper_mask = Image.open(wr_gripper_mask).convert("L")
    right_wrist_gripper_mask = np.array(right_wrist_gripper_mask)
    right_wrist_gripper_mask = (right_wrist_gripper_mask > 125).astype(int)

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
        # Set up the PyRep environment for IK in simulation
        current_directory = os.getcwd()
        DIR_PATH = os.path.join(current_directory, '../../RLBench/rlbench')
        headless = True
        pyrep = PyRep()
        pyrep.launch(os.path.join(DIR_PATH, 'task_design_bimanual.ttt'), headless=headless)
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

    # Prepare variables for optimization
    # bounds for x, y, z translations
    og_bounds = [(-magnitude_u, magnitude_u)] * 3
    normalized_bounds =[(-1, 1) for _ in og_bounds]
    # bounds for x, y, z translations for left wrist, x, y, z rotations for both wrists, and x, y, z translations for right wrist
    og_bounds_no_contacts = [(-magnitude_u, magnitude_u)] * 3 + [(x_euler_range[0], x_euler_range[1]), (y_euler_range[0], y_euler_range[1]), (z_euler_range[0], z_euler_range[1])] * 2 + [(-magnitude_u, magnitude_u)] * 3
    normalized_bounds_no_contacts =[(-1, 1) for _ in og_bounds_no_contacts]

    generation_data = []
    folders = sorted(list(data.keys()))
    image_index = 0
    save_robot_traj_visualization = args.save_robot_traj_visualization
    for ep_num, folder in enumerate(folders):
        print(f'Episode {ep_num}')
        if args.sfm_method == "orbslam_bimanual":
            left_folder_data = data[folder]['dct_left']
            right_folder_data = data[folder]['dct_right']
            gripper_data = data[folder]['dct_gripper']
            left_gripper_matrices = data[folder]['dct_left_gripper_matrices']
            right_gripper_matrices = data[folder]['dct_right_gripper_matrices']
            num_imgs = len(left_folder_data['imgs'])
        else:
            raise NotImplementedError

        # save left and right wrist images to temporay folders for SAM2
        left_wrist_images_folder = '/tmp/biaug/left_wrist_images'
        right_wrist_images_folder = '/tmp/biaug/right_wrist_images'
        create_folder_if_not_exists(left_wrist_images_folder)
        create_folder_if_not_exists(right_wrist_images_folder)
        frame_indices = []
        for frame_idx in range(0, num_imgs, args.every_x_frame):
            left_full_img_path = os.path.join(folder, left_folder_data['imgs'][frame_idx])
            right_full_img_path = os.path.join(folder, right_folder_data['imgs'][frame_idx])
            left_image = Image.open(left_full_img_path)
            left_image.save(f"{left_wrist_images_folder}/{frame_idx}.jpg")
            right_image = Image.open(right_full_img_path)
            right_image.save(f"{right_wrist_images_folder}/{frame_idx}.jpg")
            frame_indices.append(frame_idx)

        # get gripper masks for left and right wrists using SAM2
        left_gripper_video_segments, non_dilated_left_gripper_video_segments = get_gripper_masks(predictor, left_wrist_images_folder, left_wrist_gripper_mask, frame_indices, which_arm="left")
        right_gripper_video_segments, non_dilated_right_gripper_video_segments = get_gripper_masks(predictor, right_wrist_images_folder, right_wrist_gripper_mask, frame_indices, which_arm="right")

        left_wrist_traj, left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj, contacts_list = [], [], [], [], []
        for idx, frame_idx in enumerate(range(0, num_imgs, args.every_x_frame)):
            if args.sfm_method == "orbslam_bimanual":
                # if frame_idx == 0:            # for debugging
                #     print(left_full_img_path) # for debugging
                left_full_img_path = os.path.join(folder, left_folder_data['imgs'][frame_idx])
                right_full_img_path = os.path.join(folder, right_folder_data['imgs'][frame_idx])
                if gripper_data is not None:
                    gripper_key = left_folder_data['imgs'][frame_idx].split('_')[-1].split('.')[0]
            else:
                raise NotImplementedError
            
            if args.load_depth_npy:
                left_wrist_depth_img_path = f'{left_full_img_path.split(".jpg")[0]}_depth.npy'
                right_wrist_depth_img_path = f'{right_full_img_path.split(".jpg")[0]}_depth.npy'
            else:
                left_wrist_depth_img_path = f'{left_full_img_path.split(".jpg")[0]}_depth.png'
                right_wrist_depth_img_path = f'{right_full_img_path.split(".jpg")[0]}_depth.png'
            if args.no_opt:
                # contact is always false because we don't want to use constrained optimization for contact-rich states
                contact = False
            else:
                contact = determine_if_contact_occurred(args.real_robot, left_wrist_depth_img_path, right_wrist_depth_img_path, data[folder]['depth_info'], idx, frame_idx, left_gripper_video_segments, right_gripper_video_segments, non_dilated_left_gripper_video_segments, non_dilated_right_gripper_video_segments, args.load_depth_npy, left_full_img_path, right_full_img_path, ep_num=ep_num)
            # print(f'idx {idx}, contact {contact}') # for debugging

            if args.sfm_method == "orbslam_bimanual":
                left_orig_img_copy_path = os.path.join(output_root, 'images', "left_%09d_o.png" % image_index)
                right_orig_img_copy_path = os.path.join(output_root, 'images', "right_%09d_o.png" % image_index)
            else:
                raise NotImplementedError

            if args.sfm_method == "orbslam_bimanual":
                # bimanual implementation
                for i in range (args.mult):
                    opt_failures = []
                    left_output_img_path = os.path.join(output_root, "images", "left_%09d.png" % image_index)
                    right_output_img_path = os.path.join(output_root, "images", "right_%09d.png" % image_index)
                    
                    debug_results_dict = None
                    if contact:
                        current_time = time.time()
                        # when contacts occur, we want to handle the situation differently using constrained optimization
                        if args.real_robot:
                            add_args = (
                                env_swift,
                                left_robot,
                                right_robot,
                                og_bounds,
                                left_folder_data['poses_orig'][frame_idx],
                                right_folder_data['poses_orig'][frame_idx],
                                frame_idx,
                                magnitude_l,
                                x_euler_range,
                                y_euler_range,
                                z_euler_range,
                                left_gripper_matrices['gripper'][frame_idx],
                                right_gripper_matrices['gripper'][frame_idx],
                            )
                            result = dual_annealing(
                                func=objective_function_real_robot,
                                bounds=normalized_bounds,
                                args=add_args,
                                x0=[0] * 3,
                                callback=early_termination_callback,
                            )
                        else:
                            add_args = (
                                robot,
                                og_bounds,
                                left_folder_data['poses_orig'][frame_idx],
                                right_folder_data['poses_orig'][frame_idx],
                                frame_idx,
                                magnitude_l,
                                x_euler_range,
                                y_euler_range,
                                z_euler_range,
                                left_gripper_matrices['gripper'][frame_idx],
                                right_gripper_matrices['gripper'][frame_idx],
                            )
                            result = dual_annealing(
                                func=objective_function,
                                bounds=normalized_bounds,
                                args=add_args,
                                x0=[0] * 3,
                                callback=early_termination_callback,
                            )
                        opt_result = unnormalize(result.x, og_bounds)
                        if result.fun <= MINIMUM_OBJ_COST:
                            result.success = True
                            
                        if not result.success:
                            opt_failures.append(f'{ep_num}_{frame_idx}')
                        # _, debug_results_dict = objective_function(result.x, *add_args, debug_results=True)
                        
                        magnitude_left = 0.0
                        magnitude_right = 0.0
                        transformation_left = np.eye(4)
                        transformation_left[:3, 3] = opt_result[:3]
                        rot_euler_left = np.array([0, 0, 0])
                        transformation_left = np.matmul(transformation_left, pre_conversion)
                        transformation_left = np.matmul(np.linalg.inv(pre_conversion),transformation_left)

                        transformation_right = np.eye(4)
                        transformation_right[:3, 3] = opt_result[:3]
                        rot_euler_right = np.array([0, 0, 0])
                        transformation_right = np.matmul(transformation_right, pre_conversion)
                        transformation_right = np.matmul(np.linalg.inv(pre_conversion),transformation_right)
                    else:
                        if args.ik_for_non_contacts:
                            add_args = (
                                robot,
                                og_bounds_no_contacts,
                                left_folder_data['poses_orig'][frame_idx],
                                right_folder_data['poses_orig'][frame_idx],
                                frame_idx,
                                magnitude_l,
                                x_euler_range,
                                y_euler_range,
                                z_euler_range,
                                left_gripper_matrices['gripper'][frame_idx],
                                right_gripper_matrices['gripper'][frame_idx],
                            )
                            result = dual_annealing(
                                func=objective_function_no_contacts,
                                bounds=normalized_bounds_no_contacts,
                                args=add_args,
                                x0=[0] * 12,
                                callback=early_termination_callback,
                            )
                            opt_result = unnormalize(result.x, og_bounds_no_contacts)
                            if result.fun <= MINIMUM_OBJ_COST:
                                result.success = True
                            
                            if not result.success:
                                print(f'Constrained optimization failed at idx {idx}')

                            magnitude_left = 0.0
                            magnitude_right = 0.0
                            transformation_left = np.eye(4)
                            transformation_left[:3, 3] = opt_result[:3]
                            transformation_left[:3,:3] = scir.from_euler('xyz', opt_result[3:6], degrees=True).as_matrix()
                            rot_euler_left = np.array(opt_result[3:6])
                            transformation_left = np.matmul(transformation_left, pre_conversion)
                            transformation_left = np.matmul(np.linalg.inv(pre_conversion),transformation_left)

                            transformation_right = np.eye(4)
                            transformation_right[:3, 3] = opt_result[9:]
                            transformation_right[:3,:3] = scir.from_euler('xyz', opt_result[6:9], degrees=True).as_matrix()
                            rot_euler_right = np.array(opt_result[6:9])
                            transformation_right = np.matmul(transformation_right, pre_conversion)
                            transformation_right = np.matmul(np.linalg.inv(pre_conversion),transformation_right)
                        else:
                            # left wrist
                            direction_left = np.random.randn(3)
                            direction_left /= np.linalg.norm(direction_left)
                            # Sample a random magnitude
                            magnitude_left = np.random.uniform(magnitude_l,magnitude_u)
                            # Create a translation vector with the random direction and magnitude
                            translation_left = direction_left * magnitude_left
                            transformation_left = np.eye(4)
                            transformation_left[:3, 3] = translation_left
                            
                            if args.sample_rotation:
                                if x_euler_range is not None:
                                    x_euler_left = np.random.uniform(*x_euler_range)
                                else:
                                    x_euler_left = 0
                                if y_euler_range is not None:
                                    y_euler_left = np.random.uniform(*y_euler_range)
                                else:
                                    y_euler_left = 0
                                if z_euler_range is not None:
                                    z_euler_left = np.random.uniform(*z_euler_range)
                                else:
                                    z_euler_left = 0
                                rot_euler_left = np.array([x_euler_left, y_euler_left, z_euler_left])
                                rot_left = scir.from_euler('xyz', rot_euler_left, degrees=True).as_matrix()
                                transformation_left[:3,:3] = rot_left                
                            transformation_left = np.matmul(transformation_left, pre_conversion)
                            transformation_left = np.matmul(np.linalg.inv(pre_conversion),transformation_left)

                            # right wrist
                            direction_right = np.random.randn(3)
                            direction_right /= np.linalg.norm(direction_right)
                            # Sample a random magnitude between 0 and 5
                            magnitude_right = np.random.uniform(magnitude_l,magnitude_u)
                            # Create a translation vector with the random direction and magnitude
                            translation_right = direction_right * magnitude_right
                            transformation_right = np.eye(4)
                            transformation_right[:3, 3] = translation_right
                            
                            if args.sample_rotation:
                                if x_euler_range is not None:
                                    x_euler_right = np.random.uniform(*x_euler_range)
                                else:
                                    x_euler_right = 0
                                if y_euler_range is not None:
                                    y_euler_right = np.random.uniform(*y_euler_range)
                                else:
                                    y_euler_right = 0
                                if z_euler_range is not None:
                                    z_euler_right = np.random.uniform(*z_euler_range)
                                else:
                                    z_euler_right = 0
                                rot_euler_right = np.array([x_euler_right, y_euler_right, z_euler_right])
                                rot_right = scir.from_euler('xyz', rot_euler_right, degrees=True).as_matrix()
                                transformation_right[:3,:3] = rot_right               
                            transformation_right = np.matmul(transformation_right, pre_conversion)
                            transformation_right = np.matmul(np.linalg.inv(pre_conversion),transformation_right)
                    
                    this_data = {
                        "left_img": left_full_img_path,
                        "right_img": right_full_img_path,
                        "left_orig_img_copy_path" : left_orig_img_copy_path,
                        "right_orig_img_copy_path" : right_orig_img_copy_path,
                        "wl_focal_y": left_folder_data["focal_y"],
                        "wr_focal_y": right_folder_data["focal_y"],
                        "left_output": left_output_img_path,
                        "right_output": right_output_img_path,
                        "magnitude_left" : magnitude_left,
                        "transformation_left": transformation_left.tolist(),
                        "magnitude_right" : magnitude_right,
                        "transformation_right": transformation_right.tolist(),
                        "opt_failures": opt_failures,
                    }
                    if args.sample_rotation:
                        this_data["rot_euler_left"] = rot_euler_left.astype(float).tolist()
                        this_data["rot_euler_right"] = rot_euler_right.astype(float).tolist()
                    if gripper_data is not None:
                        this_data["gripper_data"] = gripper_data[gripper_key].tolist()
                    if debug_results_dict is not None:
                        this_data["left_arm_new_joint_positions"] = debug_results_dict['left_arm_new_joint_positions'].tolist()
                        this_data["right_arm_new_joint_positions"] = debug_results_dict['right_arm_new_joint_positions'].tolist()
                    if len(opt_failures) > 0:
                        print(f'Optimization failed for {opt_failures}')
                    generation_data.append(this_data)
                    image_index += 1
            else:
                raise NotImplementedError
            
            if save_robot_traj_visualization:
                left_wrist_traj.append((
                    (left_folder_data['poses_orig'][frame_idx][0, -1], left_folder_data['poses_orig'][frame_idx][1, -1], left_folder_data['poses_orig'][frame_idx][2, -1]),
                    left_folder_data['poses_orig'][frame_idx][:3, :3]
                ))
                right_wrist_traj.append((
                    (right_folder_data['poses_orig'][frame_idx][0, -1], right_folder_data['poses_orig'][frame_idx][1, -1], right_folder_data['poses_orig'][frame_idx][2, -1]),
                    right_folder_data['poses_orig'][frame_idx][:3, :3]
                ))

                transformation_left = from_blender_frame(transformation_left)
                left_arm_left_wrist_cam_aug_pose = np.matmul(transformation_left, left_folder_data['poses_orig'][frame_idx])
                left_wrist_aug_traj.append((
                    (left_arm_left_wrist_cam_aug_pose[0, -1], left_arm_left_wrist_cam_aug_pose[1, -1], left_arm_left_wrist_cam_aug_pose[2, -1]),
                    left_arm_left_wrist_cam_aug_pose[:3, :3],
                ))
                
                transformation_right = from_blender_frame(transformation_right)
                right_arm_right_wrist_cam_aug_pose = np.matmul(transformation_right, right_folder_data['poses_orig'][frame_idx])
                right_wrist_aug_traj.append((
                    (right_arm_right_wrist_cam_aug_pose[0, -1], right_arm_right_wrist_cam_aug_pose[1, -1], right_arm_right_wrist_cam_aug_pose[2, -1]),
                    right_arm_right_wrist_cam_aug_pose[:3, :3],
                ))

                contacts_list.append(contact)

        if save_robot_traj_visualization:
            visualize_robot_trajectory(left_wrist_traj, left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj, contacts_list, ep_num)

    print("Total number of images:", len(generation_data))
    out_file = os.path.join(output_root, 'data.json')
    if os.path.exists(out_file):
        print(f"{out_file} file exists")
    else:
        with open(out_file, 'w') as f: json.dump(generation_data, f)
        print("Written to: ", out_file)
    
    if args.save_print_to_file:
        # Restore standard output to the console
        sys.stdout = sys.__stdout__
        # Close the file
        print_output_file.close()
        print('Print output saved to:', print_output_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sfm_method', type=str, choices=['colmap', 'grabber_orbslam', 'orbslam_bimanual'])
    parser.add_argument("--data_folders", type=str, nargs="+", default=[])
    parser.add_argument("--focal_lengths", type=float, nargs="+", default=[])
    parser.add_argument("--real_wl_focal_length", type=float, default=None)
    parser.add_argument("--real_wr_focal_length", type=float, default=None)
    parser.add_argument("--real_robot", action="store_true", help="Whether to do inference on real robot data")
    parser.add_argument("--image_heights", type=float, nargs="+", default=[])
    parser.add_argument('--output_root', type=str, help="output folder")
    parser.add_argument('--suffix', default=None, help="suffix to add to output_root, if you want to generate multiple versions")
    parser.add_argument('--sample_rotation',action='store_true')
    parser.add_argument('--every_x_frame', type=int, default=10, help="Generate augmenting images of every every_x_frame-th frame. If the demonstrations are recorded at high frame-rate (e.g. above 5fps), nearby frames are very similar, and there are too many frames in each trajectory, so it is not necessary to generate augmenting samples for every frame.")
    parser.add_argument('--mult',type=int,default=3,help='Number of augmenting images to generate per input frame')
    parser.add_argument('--save_print_to_file',action='store_true')
    parser.add_argument('--seed', type=int, default=0, help="Random seed number.")
    parser.add_argument('--save_robot_traj_visualization',action='store_true')
    parser.add_argument('--load_depth_npy',action='store_true')
    parser.add_argument('--ik_for_non_contacts',action='store_true')
    parser.add_argument('--no_opt', action='store_true', help="No constrained optimization for contact-rich states")

    args = parser.parse_args()

    main(args)