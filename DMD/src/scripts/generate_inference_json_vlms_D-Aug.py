import os
import random
import sys
import argparse
import pathlib
import json
import numpy as np
from scipy.spatial.transform import Rotation as scir
import itertools

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



# add
CUTOFF_NUM = 50



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

def reset_bimanual(pyrep, robot, initial_robot_state, start_arm_joint_pos, starting_gripper_joint_pos):
    for arm, gripper in initial_robot_state:        
        pyrep.set_configuration_tree(arm)
        pyrep.set_configuration_tree(gripper)
    
    robot.right_arm.set_joint_positions(start_arm_joint_pos[0], disable_dynamics=True)
    robot.right_gripper.set_joint_positions(starting_gripper_joint_pos[0], disable_dynamics=True)

    robot.left_arm.set_joint_positions(start_arm_joint_pos[1], disable_dynamics=True)
    robot.left_gripper.set_joint_positions(starting_gripper_joint_pos[1], disable_dynamics=True)

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

def world_perturbation_to_camera_frame(world_transform, camera_pose):
    """
    Convert a world-frame left-multiplied perturbation into an equivalent
    camera-local right-multiplied perturbation.
    """
    # transform because of newly defined xyz
    C = np.array([
        [0,  1, 0, 0],
        [1, 0, 0, 0],
        [0,  0, -1, 0],
        [0,  0, 0, 1],
            ], dtype=float)
    camera_pose = C @ camera_pose 

    local_transform = np.linalg.inv(camera_pose) @ world_transform @ camera_pose
    local_delta_xyz = local_transform[:3, 3]
    local_euler_xyz = scir.from_matrix(local_transform[:3, :3]).as_euler('xyz', degrees=True)
    return local_delta_xyz, local_euler_xyz

def get_camera_pose_from_extrinsics(camera_extrinsics, image_path, fallback_pose):
    """Get a valid 4x4 camera pose matrix for the current frame."""
    image_name = os.path.basename(image_path)
    matrix = camera_extrinsics.get(image_name)
    if matrix is None:
        print(f"Warning: camera_extrinsics missing key {image_name}, fallback to poses_orig")
        return fallback_pose

    camera_pose = np.asarray(matrix, dtype=np.float64)
    if camera_pose.shape != (4, 4):
        print(f"Warning: invalid camera_extrinsics shape for {image_name}: {camera_pose.shape}, fallback to poses_orig")
        return fallback_pose
    return camera_pose

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
    
    # 使用 zip_longest，不补全默认设为 None
    for l_pose, l_a_pose, r_pose, r_a_pose, contact in itertools.zip_longest(
        left_wrist_traj, left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj, contacts_list, fillvalue=None):
        
        # 1. 始终处理原始轨迹 (如果存在)
        if l_pose is not None:
            l_position, l_R = l_pose
            l_positions.append(l_position)
            add_line_segment_for_each_basis(index, traces, l_position, l_R, prefix='Org L')
            
        if r_pose is not None:
            r_position, r_R = r_pose
            r_positions.append(r_position)
            add_line_segment_for_each_basis(index, traces, r_position, r_R, prefix='Org R')

        # 2. 如果当前步存在增强轨迹（以及联动的contact），则正常填充
        if l_a_pose is not None and r_a_pose is not None:
            l_a_position, l_a_R = l_a_pose
            r_a_position, r_a_R = r_a_pose
            
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
        raise NotImplementedError("Real robot setup is not implemented yet")
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


        # new
        # Here we assume all of the tasks use the same range for x, y, z rotations and magnitudes
        x_euler_range = [-0, 0]
        y_euler_range = [-0, 0]
        z_euler_range = [-0, 0]
        magnitude_l, magnitude_u = 0.08,0.15

        episode_start_delta_X_left = random.uniform(magnitude_l, magnitude_u)*random.choice([-1, 1])
        episode_start_delta_Y_left = random.uniform(magnitude_l, magnitude_u)*random.choice([-1, 1])
        episode_start_delta_Z_left = random.uniform(magnitude_l, magnitude_u)*random.choice([-1, 1])
        episode_start_x_euler_left = random.uniform(x_euler_range[0], x_euler_range[1])
        episode_start_y_euler_left = random.uniform(y_euler_range[0], y_euler_range[1])
        episode_start_z_euler_left = random.uniform(z_euler_range[0], z_euler_range[1])

        episode_start_delta_X_right = random.uniform(magnitude_l, magnitude_u)*random.choice([-1, 1])
        episode_start_delta_Y_right = random.uniform(magnitude_l, magnitude_u)*random.choice([-1, 1])
        episode_start_delta_Z_right = random.uniform(magnitude_l, magnitude_u)*random.choice([-1, 1])
        episode_start_x_euler_right = random.uniform(x_euler_range[0], x_euler_range[1])
        episode_start_y_euler_right = random.uniform(y_euler_range[0], y_euler_range[1])
        episode_start_z_euler_right = random.uniform(z_euler_range[0], z_euler_range[1])

        if args.sfm_method == "orbslam_bimanual":
            left_folder_data = data[folder]['dct_left']
            right_folder_data = data[folder]['dct_right']
            gripper_data = data[folder]['dct_gripper']
            left_gripper_matrices = data[folder]['dct_left_gripper_matrices']
            right_gripper_matrices = data[folder]['dct_right_gripper_matrices']
            num_imgs = min(len(left_folder_data['imgs']), args.cutoff_num)

            camera_extrinsics_path = os.path.join(folder, 'camera_extrinsics.json')
            camera_extrinsics = {}
            if os.path.exists(camera_extrinsics_path):
                with open(camera_extrinsics_path, 'r') as f:
                    camera_extrinsics = json.load(f)
            else:
                print(f"Warning: no camera_extrinsics.json at {camera_extrinsics_path}, fallback to poses_orig")
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
            left_image.save(f"{left_wrist_images_folder}/{frame_idx}.png")
            right_image = Image.open(right_full_img_path)
            right_image.save(f"{right_wrist_images_folder}/{frame_idx}.png")
            frame_indices.append(frame_idx)

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
            
            if args.no_opt:
                # contact is always false because we don't want to use constrained optimization for contact-rich states
                contact = False
            else:
                raise NotImplementedError

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
                        raise NotImplementedError
                    else:
                        if args.ik_for_non_contacts:
                            raise NotImplementedError
                        else:
                            # left wrist
                            # 根据当前帧进度计算权重，随着帧数增加，偏移量逐渐衰减至 0
                            weight = 1.0 - (frame_idx / max(1, num_imgs - 1))
                            
                            delta_x_left = episode_start_delta_X_left * weight
                            delta_y_left = episode_start_delta_Y_left * weight
                            delta_z_left = episode_start_delta_Z_left * weight
                            
                            translation_left = np.array([delta_x_left, delta_y_left, delta_z_left])
                            magnitude_left = np.linalg.norm(translation_left) # 记录magnitude如果有其他地方用到

                            transformation_left = np.eye(4)
                            transformation_left[:3, 3] = translation_left
                            
                            if args.sample_rotation:
                                x_euler_left = episode_start_x_euler_left * weight if x_euler_range is not None else 0
                                y_euler_left = episode_start_y_euler_left * weight if y_euler_range is not None else 0
                                z_euler_left = episode_start_z_euler_left * weight if z_euler_range is not None else 0
                                
                                rot_euler_left = np.array([x_euler_left, y_euler_left, z_euler_left])
                                rot_left = scir.from_euler('xyz', rot_euler_left, degrees=True).as_matrix()
                                transformation_left[:3,:3] = rot_left  
                                              
                            transformation_left = np.matmul(transformation_left, pre_conversion)
                            transformation_left = np.matmul(np.linalg.inv(pre_conversion), transformation_left)

                            # right wrist
                            # 根据当前帧进度计算权重，随着帧数增加，偏移量逐渐衰减至 0
                            weight = 1.0 - (frame_idx / max(1, num_imgs - 1))
                            
                            delta_x_right = episode_start_delta_X_right * weight
                            delta_y_right = episode_start_delta_Y_right * weight
                            delta_z_right = episode_start_delta_Z_right * weight
                            
                            translation_right = np.array([delta_x_right, delta_y_right, delta_z_right])
                            magnitude_right = np.linalg.norm(translation_right) # 记录magnitude
                            
                            transformation_right = np.eye(4)
                            transformation_right[:3, 3] = translation_right
                            
                            if args.sample_rotation:
                                x_euler_right = episode_start_x_euler_right * weight if x_euler_range is not None else 0
                                y_euler_right = episode_start_y_euler_right * weight if y_euler_range is not None else 0
                                z_euler_right = episode_start_z_euler_right * weight if z_euler_range is not None else 0
                                
                                rot_euler_right = np.array([x_euler_right, y_euler_right, z_euler_right])
                                rot_right = scir.from_euler('xyz', rot_euler_right, degrees=True).as_matrix()
                                transformation_right[:3,:3] = rot_right               
                            transformation_right = np.matmul(transformation_right, pre_conversion)
                            transformation_right = np.matmul(np.linalg.inv(pre_conversion),transformation_right)

                    left_cam_delta_xyz, left_cam_euler_xyz = world_perturbation_to_camera_frame(
                        transformation_left,
                        get_camera_pose_from_extrinsics(
                            camera_extrinsics,
                            left_folder_data['imgs'][frame_idx],
                            left_folder_data['poses_orig'][frame_idx],
                        ),
                    )
                    right_cam_delta_xyz, right_cam_euler_xyz = world_perturbation_to_camera_frame(
                        transformation_right,
                        get_camera_pose_from_extrinsics(
                            camera_extrinsics,
                            right_folder_data['imgs'][frame_idx],
                            right_folder_data['poses_orig'][frame_idx],
                        ),
                    )
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
                        "camera_delta_xyz_left": left_cam_delta_xyz.astype(float).tolist(),
                        "camera_euler_xyz_left": left_cam_euler_xyz.astype(float).tolist(),
                        "magnitude_right" : magnitude_right,
                        "transformation_right": transformation_right.tolist(),
                        "camera_delta_xyz_right": right_cam_delta_xyz.astype(float).tolist(),
                        "camera_euler_xyz_right": right_cam_euler_xyz.astype(float).tolist(),
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
            
            visualize_interval = args.visualize_interval
            if save_robot_traj_visualization and frame_idx % visualize_interval == 0:
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
            for idx, frame_idx in enumerate(range(num_imgs, len(left_folder_data['imgs'])-1, 1)):
                if frame_idx % visualize_interval == 0:
                    left_wrist_traj.append((
                        (left_folder_data['poses_orig'][frame_idx][0, -1], left_folder_data['poses_orig'][frame_idx][1, -1], left_folder_data['poses_orig'][frame_idx][2, -1]),
                        left_folder_data['poses_orig'][frame_idx][:3, :3]
                    ))
                    right_wrist_traj.append((
                        (right_folder_data['poses_orig'][frame_idx][0, -1], right_folder_data['poses_orig'][frame_idx][1, -1], right_folder_data['poses_orig'][frame_idx][2, -1]),
                        right_folder_data['poses_orig'][frame_idx][:3, :3]
                    ))
                    contacts_list.append(False)
            visualize_robot_trajectory(left_wrist_traj, left_wrist_aug_traj, right_wrist_traj, right_wrist_aug_traj, contacts_list, ep_num)

    print("Total number of images:", len(generation_data))
    out_file = os.path.join(output_root, 'data.json')

    print("Writing to:", out_file)
    if os.path.exists(out_file):
        print(f"{out_file} file exists, overwriting.")
    with open(out_file, 'w') as f:
        json.dump(generation_data, f)
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
    parser.add_argument('--cutoff_num', type=int, default=50, help="Only first x pictures.")
    parser.add_argument('--mult',type=int,default=3,help='Number of augmenting images to generate per input frame')
    parser.add_argument('--save_print_to_file',action='store_true')
    parser.add_argument('--seed', type=int, default=0, help="Random seed number.")
    parser.add_argument('--save_robot_traj_visualization',action='store_true')
    parser.add_argument('--load_depth_npy',action='store_true')
    parser.add_argument('--ik_for_non_contacts',action='store_true')
    parser.add_argument('--no_opt', action='store_true', help="No constrained optimization for contact-rich states")
    parser.add_argument('--visualize_interval', type=int, default=1, help="Visualize trajectory, points.")
    

    args = parser.parse_args()

    main(args)