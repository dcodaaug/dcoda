import gc
import logging
import os
import sys

import peract_config

import hydra
import numpy as np
import torch
import pandas as pd
import time
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.action_mode import BimanualJointPositionActionMode
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.action_modes.arm_action_modes import BimanualJointPosition, JointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, BimanualGripperJointPosition
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import BimanualObservation, UnimanualObservationData

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from helpers.clip.core.clip import tokenize

from helpers import utils
from helpers import observation_utils

from yarr.utils.rollout_generator import RolloutGenerator
import torch.multiprocessing as mp

from agents import agent_factory
from abc import abstractmethod
from typing import Dict, Protocol
import urx as urx
import multiprocessing
import threading
from PIL import Image
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation as R
from multiprocessing import Value
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import cv2

TARGET_IMAGE_SIZE = 128
ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc', 'left', 'right']
DISTANCE_BETWEEN_LEFT_RIGHT_ARMS = 1.22 # meters

def slow_servoj_move(robot, q_target, steps=100):
    q_current = robot.robot.getj()        
    for i in range(steps):
        q_interp = [(1 - i/steps) * q_current[j] + (i/steps) * q_target[j] for j in range(6)]
        try:
            robot.robot.servoj_robot(q_interp, 0.001, 0.003, 0.3, 0.2, 300)
        except:
            pass

def execute_gripper_action(robot, gripper_action):
    if gripper_action > 0.5:
        # close gripper
        print(f'action {gripper_action} close gripper')
        robot.robot.set_digital_out(8, True)
        robot.robot.set_digital_out(9, False)
        time.sleep(0.05)
    else:
        print(f'action {gripper_action} open gripper')
        robot.robot.set_digital_out(8, False)
        robot.robot.set_digital_out(9, False)
        time.sleep(0.05)

class Robot(Protocol):
    """Robot protocol 
    Copied from gello_software_v2

    A protocol for a robot that can be controlled.
    """

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        raise NotImplementedError

    @abstractmethod
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        raise NotImplementedError
class URXRobot(Robot):
    """
    Copied from gello_software_v2
    A class representing a UR robot.
    """

    def __init__(self, robot_ip: str = "192.10.0.11", no_gripper: bool = False):
        # import rtde_control
        # import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            # self.robot = rtde_control.RTDEControlInterface(robot_ip)
            self.robot = urx.Robot(robot_ip)
            # print('done')
        except Exception as e:
            print(e)
            print(robot_ip)

        # self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            print('Sending activation sequence...')
            self.robot.set_tool_voltage(24)
            self.robot.set_digital_out(8, False)
            self.robot.set_digital_out(9, False)
            time.sleep(0.05)

            self.robot.set_digital_out(8, True)
            time.sleep(0.05)

            self.robot.set_digital_out(8, False)
            time.sleep(0.05)

            self.robot.set_digital_out(9, True)
            time.sleep(0.05)

            self.robot.set_digital_out(9, False)
            time.sleep(0.05)


        [print("connect") for _ in range(4)]

        self._free_drive = False
        # self.robot.set_freedrive(True)
        # self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        # time.sleep(0.01)
        # gripper_pos = self.gripper.get_current_position()
        # gripper_pos = 0
        # assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        dout0 = self.robot.get_digital_out(8)
        dout1 = self.robot.get_digital_out(9)
        if dout0 > 0.5 or dout1 > 0.5:
            gripper_pos = 1.0 # close
        else:
            gripper_pos = 0.0 # open
        assert 0 <= gripper_pos <= 1, "Gripper position must be between 0 and 1"
        return gripper_pos

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # robot_joints = self.r_inter.getActualQ()
        robot_joints = np.array(self.robot.getj()) 
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5 
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100
        t=0.5

        # velocity = 0.1
        # acceleration = 0.1
        # dt = 1.0 / 500  # 2ms
        # lookahead_time = 0.2
        # gain = 100
        # t=0.5

        robot_joints = joint_state[:6]
        # t_start = self.robot.initPeriod()
        # t_start = time.time()

        # gripper control
        if self._use_gripper:
            gripper_pos = joint_state[-1] # range [0, 1]
            if gripper_pos > 0.5: # close
                self.robot.set_digital_out(8, True)
                self.robot.set_digital_out(9, False)
            else: # open
                self.robot.set_digital_out(8, False)
                self.robot.set_digital_out(9, False)
        
        # arm control
        self.robot.servoj_robot(
            robot_joints, velocity, acceleration, t, lookahead_time, gain, wait=False,
            # robot_joints, velocity, acceleration, t, lookahead_time, gain, wait=True,
        )
        # self.robot.servoj_robot(
        #     tjoints=robot_joints, acc=0.01, vel=0.01, t=0.1, lookahead_time=0.2, gain=100, wait=False, relative=False, threshold=None
        # )
        # self.robot.set_pos(
        #     robot_joints,  acceleration, velocity
        # )
        # print("commanded joint state")
        # self.robot.waitPeriod(t_start)
        # elapsed_time = time.time() - t_start
        # while elapsed_time < dt:
        #     time.sleep(0.001)  # Sleep for a short time (1 ms) to avoid busy-waiting
        #     elapsed_time = time.time() - t_start


    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.set_freedrive(True)
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.set_freedrive(False)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        
        tip_pos = self.robot.getl()[:3]
        tip_orientation = self.robot.get_orientation()
        euler_angle = tip_orientation.to_euler('ZYX') # roll-pitch-yaw
        quat = self.get_quaternion_from_euler(euler_angle[0], euler_angle[1], euler_angle[2])
        pos_quat = np.array([tip_pos[0], tip_pos[1], tip_pos[2], quat[0], quat[1], quat[2], quat[3]]) # x, y, z, qx, qy, qz, qw

        if self._use_gripper:
            gripper_pos = np.array([joints[-1]])
        else:
            gripper_pos = np.zeros(1)
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }
    
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
        return [qx, qy, qz, qw]
class BimanualRobot(Robot):
    """
    Copied from gello_software_v2
    """
    def __init__(self, robot_l: Robot, robot_r: Robot):
        self._robot_l = robot_l
        self._robot_r = robot_r

    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._robot_l.command_joint_state(joint_state[: self._robot_l.num_dofs()])
        self._robot_r.command_joint_state(joint_state[self._robot_l.num_dofs() :])

    def get_observations(self) -> Dict[str, np.ndarray]:
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs
class CameraDriver(Protocol):
    """Camera protocol.

    A protocol for a camera driver. This is used to abstract the camera from the rest of the code.
    """

    def read(
        self,
        img_size = None,
    ):
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """
class RealSenseCamera(CameraDriver):
    """
    Copied from gello_software_v2
    """
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id = None, flip = False):
        import pyrealsense2 as rs

        self._device_id = device_id

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                if device_id == dev.get_info(rs.camera_info.serial_number):
                    dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        self.enable_depth = True
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline_cfg = self._pipeline.start(config)
        self._rs_stream = rs.stream
        self._flip = flip

    def read(
        self,
        img_size = None,  # farthest: float = 0.12
    ):
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            if self.enable_depth:
                depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            if self.enable_depth:
                depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            if self.enable_depth:
                depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            if self.enable_depth:
                depth = depth[:, :, None]
        if not self.enable_depth:
            depth = np.array([])

        # this is used to visualize the current camera image during teleop, so I know where to move the arms to the starting positions
        # cv2.imwrite(f"/home/arthurliu/Documents/peract_bimanual/data/real_sense_{self._device_id}.png", image)
        return image, depth

    def format_cam_intrinsics(self, dict):
        return  np.array([
                    [dict['fx'], 0, dict['cx']],
                    [0, dict['fy'], dict['cy']],
                    [0, 0, 1],
                ])

    def get_wrist_left_intrinsics(self):
        """
        Got this info from python gello/cameras/realsense_camera.py 
        """
        return self.format_cam_intrinsics({
            'width': 640,
            'height': 480,
            'fx': 599.4510498046875,
            'fy': 599.4510498046875,
            'cx': 321.3397216796875,
            'cy': 236.63348388671875,
        })
    
    def get_wrist_right_intrinsics(self):
        """
        Got this info from python gello/cameras/realsense_camera.py 
        """
        return self.format_cam_intrinsics({
            'width': 640,
            'height': 480,
            'fx': 598.1443481445312,
            'fy': 598.1443481445312,
            'cx': 325.3073425292969,
            'cy': 247.422119140625,
        })

    def get_front_intrinsics(self):
        """
        Got this info from python gello/cameras/realsense_camera.py 
        """
        return self.format_cam_intrinsics({
            'width': 640,
            'height': 480,
            'fx': 592.737060546875,
            'fy': 592.737060546875,
            'cx': 320.4691162109375,
            'cy': 246.06141662597656,
        })

class RobotEnv():
    def __init__(self, train_cfg, eval_cfg, env_config):
        self._train_cfg = train_cfg
        self.robot = None
        if train_cfg.rlbench.tasks[0] == 'coordinated_lift_ball':
            self.left_starting_state = np.array([ 1.083174  , -1.07447183, -2.23271003, -1.21964233,  1.30875644, 2.32890764])
            self.right_starting_state = np.array([-1.11736182, -2.0778797 ,  2.26183358, -1.9743213 , -1.16255898, 0.51213587])
        elif train_cfg.rlbench.tasks[0] == 'coordinated_push_box':
            self.left_starting_state = np.array([0.94735887, -1.79499494, -1.80333075, -1.26109021,  1.61209071,  2.38173996])
            self.right_starting_state = np.array([-1.00327588, -1.36185112,  1.81914982, -1.95721224, -1.52214172, 0.47910908])
        elif train_cfg.rlbench.tasks[0] == 'lift_drawer':
            self.left_starting_state = np.array([0.53897068, -1.42173197, -1.65254373, -1.67323836,  1.65733595,  3.38956017])
            self.right_starting_state = np.array([-0.64280017, -1.60397262,  1.61477863, -1.43984548, -1.50003802, -0.55196522])
        else:
            raise NotImplementedError

        # initialize camera
        left_camera_id = '151322067992'
        right_camera_id = '217222067236'
        self.wl_camera = RealSenseCamera(left_camera_id)
        self.wr_camera = RealSenseCamera(right_camera_id)
        self.wl_camera_intrinsics = self.wl_camera.get_wrist_left_intrinsics()
        self.wr_camera_intrinsics = self.wr_camera.get_wrist_right_intrinsics()
        self.front_cam = False
        if 'front' in eval_cfg.rlbench.cameras:
            self.front_cam = True
            front_camera_id = '217222063197'
            self.front_camera = RealSenseCamera(front_camera_id)
            self.front_camera_intrinsics = self.front_camera.get_front_intrinsics()

        self._include_lang_goal_in_obs = train_cfg.rlbench.include_lang_goal_in_obs
        self._time_in_state = eval_cfg.rlbench.time_in_state
        self._episode_length = eval_cfg.rlbench.episode_length
        self._observation_config = env_config[1]
        self._channels_last = False
        self._timesteps = 1
        self._i = 0
        self._lang_goal = None
        if train_cfg['rlbench']['tasks'][0] == 'coordinated_lift_ball':
            self._lang_goal = "Lift the ball"
        elif train_cfg['rlbench']['tasks'][0] == 'coordinated_push_box':
            self._lang_goal = "push the box to the edge of the mat"
        elif train_cfg['rlbench']['tasks'][0] == 'lift_drawer':
            self._lang_goal = "lift drawer"
        else:
            print('NOTE: Need to add language goal for this task...')
            raise NotImplementedError()

        self.real_world_execute_k_actions = eval_cfg.framework.real_world_execute_k_actions

        # for visualization
        self.real_world_viz = eval_cfg.framework.real_world_viz
        if self.real_world_viz:
            self.env_swift = swift.Swift() 
            self.env_swift_dt = 0.05
            self.env_swift.launch(realtime=True)
            self.env_swift_left_robot = rtb.models.URDF.UR5()
            self.env_swift_right_robot = rtb.models.URDF.UR5()
            self.env_swift_left_robot.q = self.left_starting_state
            self.env_swift_right_robot.q = self.right_starting_state
            # move to the left to create our bimanual UR5 setup
            self.env_swift_left_robot.base = sm.SE3(DISTANCE_BETWEEN_LEFT_RIGHT_ARMS, 0, 0)
            self.env_swift.add(self.env_swift_left_robot)
            self.env_swift.add(self.env_swift_right_robot)

    def initialize_robot(self, robot_left_ip, robot_right_ip):
        _robot_l = URXRobot(robot_ip=robot_left_ip)
        _robot_r = URXRobot(robot_ip=robot_right_ip)
        self.robot = BimanualRobot(_robot_l, _robot_r)
    
    def _extract_obs_bimanual(self, obs: BimanualObservation, channels_last: bool, observation_config: ObservationConfig):
        """
        Copied from yarr/envs/rlbench_env.py
        """
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}

        right_robot_state = obs.get_low_dim_data(obs.right)
        left_robot_state = obs.get_low_dim_data(obs.left)

        obs_dict = {k: v for k, v in obs_dict.items()
                if k not in ROBOT_STATE_KEYS}

        if not channels_last:
            # Swap channels from last dim to 1st dim
            obs_dict = {k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                        for k, v in obs.perception_data.items() if v is not None}
        else:
            # Add extra dim to depth data
            obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                        for k, v in obs.perception_data.items() if v is not None}
            
        if observation_config.robot_name == 'right':
            obs_dict['low_dim_state'] = right_robot_state.astype(np.float32)
            obs_dict['ignore_collisions'] = np.array([obs.right.ignore_collisions], dtype=np.float32)
        elif observation_config.robot_name == 'left':
            obs_dict['low_dim_state'] = left_robot_state.astype(np.float32)
            obs_dict['ignore_collisions'] = np.array([obs.left.ignore_collisions], dtype=np.float32)
        else:
            obs_dict['right_low_dim_state'] = right_robot_state.astype(np.float32)
            obs_dict['left_low_dim_state'] = left_robot_state.astype(np.float32)
            obs_dict['right_ignore_collisions'] = np.array([obs.right.ignore_collisions], dtype=np.float32)
            obs_dict['left_ignore_collisions'] = np.array([obs.left.ignore_collisions], dtype=np.float32)

        for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
            obs_dict[k] = v.astype(np.float16)

        for camera_name, config in observation_config.camera_configs.items():
            if config.point_cloud:
                obs_dict[f'{camera_name}_camera_extrinsics'] = obs.misc[f'{camera_name}_camera_extrinsics']
                obs_dict[f'{camera_name}_camera_intrinsics'] = obs.misc[f'{camera_name}_camera_intrinsics']
        return obs_dict

    def extract_obs_bimanual(self, obs: BimanualObservation, t=None, prev_action=None):
        """
        Copied from helpers/custom_rlbench_env.py
        """
        obs.right.joint_velocities = None
        right_grip_mat = obs.right.gripper_matrix
        right_grip_pose = obs.right.gripper_pose
        right_joint_pos = obs.right.joint_positions
        obs.right.gripper_pose = None
        obs.right.gripper_matrix = None
        obs.right.joint_positions = None

        obs.left.joint_velocities = None
        left_grip_mat = obs.left.gripper_matrix
        left_grip_pose = obs.left.gripper_pose
        left_joint_pos = obs.left.joint_positions
        obs.left.gripper_pose = None
        obs.left.gripper_matrix = None
        obs.left.joint_positions = None

        if obs.right.gripper_joint_positions is not None:
            obs.right.gripper_joint_positions = np.clip(
                obs.right.gripper_joint_positions, 0.0, 0.04
            )
            obs.left.gripper_joint_positions = np.clip(
                obs.left.gripper_joint_positions, 0.0, 0.04
            )


        obs_dict = self._extract_obs_bimanual(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            obs_dict['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0

            if "low_dim_state" in obs_dict:
                obs_dict["low_dim_state"] = np.concatenate(
                    [obs_dict["low_dim_state"], [time]]
                ).astype(np.float32)
            else:
                obs_dict["right_low_dim_state"] = np.concatenate(
                    [obs_dict["right_low_dim_state"], [time]]
                ).astype(np.float32)
                obs_dict["left_low_dim_state"] = np.concatenate(
                    [obs_dict["left_low_dim_state"], [time]]
                ).astype(np.float32)

        obs.right.gripper_matrix = right_grip_mat
        obs.right.joint_positions = right_joint_pos
        obs.right.gripper_pose = right_grip_pose
        obs.left.gripper_matrix = left_grip_mat
        obs.left.joint_positions = left_joint_pos
        obs.left.gripper_pose = left_grip_pose

        obs_dict["left_joint_positions"] = obs.left.joint_positions
        obs_dict["left_gripper_joint_positions"] = obs.left.gripper_joint_positions
        obs_dict["right_joint_positions"] = obs.right.joint_positions
        obs_dict["right_gripper_joint_positions"] = obs.right.gripper_joint_positions

        return obs_dict

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def adjust_intrinsics_with_padding_and_resize(
        self,
        intrinsics: np.ndarray,
        original_width: int,
        original_height: int,
        final_size: int
    ) -> np.ndarray:
        """
        Adjusts camera intrinsics after:
        1. Padding the image (typically the height) to make it square.
        2. Resizing the padded square image to a smaller resolution.

        Parameters:
            intrinsics (np.ndarray): Original 3x3 camera intrinsic matrix.
            original_width (int): Original image width (e.g., 640).
            original_height (int): Original image height (e.g., 480).
            final_size (int): Target final square size (e.g., 128).

        Returns:
            np.ndarray: Adjusted 3x3 intrinsic matrix.
        """
        assert intrinsics.shape == (3, 3), "Intrinsics must be a 3x3 matrix."
        assert original_width >= original_height, "Only supports padding height to match width."

        # Step 1: Adjust cy for padding (top padding shifts content down)
        pad_total = original_width - original_height
        pad_top = pad_total // 2

        new_intrinsics = intrinsics.copy()
        new_intrinsics[1, 2] += pad_top  # Adjust cy

        # Step 2: Compute scale factor from padded image to final size
        scale = final_size / original_width

        # Step 3: Scale intrinsics
        new_intrinsics[0, 0] *= scale  # fx
        new_intrinsics[1, 1] *= scale  # fy
        new_intrinsics[0, 2] *= scale  # cx
        new_intrinsics[1, 2] *= scale  # cy

        return new_intrinsics

    def pad_image(self, image, mode='constant'):
        # Use width as the desired output shape
        output_shape = (image.shape[1], image.shape[1], image.shape[2])

        # Calculate the padding sizes for the first two dimensions
        padding_top = (output_shape[0] - image.shape[0]) // 2
        padding_bottom = output_shape[0] - image.shape[0] - padding_top
        padding_left = (output_shape[1] - image.shape[1]) // 2
        padding_right = output_shape[1] - image.shape[1] - padding_left

        # Apply the padding
        if mode == 'constant':
            padded_image = np.pad(image, 
                                    ((padding_top, padding_bottom), 
                                    (padding_left, padding_right), 
                                    (0, 0)), 
                                    mode=mode, constant_values=0)
        else:
            padded_image = np.pad(image, 
                                ((padding_top, padding_bottom), 
                                (padding_left, padding_right), 
                                (0, 0)), 
                                mode=mode)
        return padded_image

    def get_observation(self):
        """
        Based on peract/helpers/custom_rlbench_env_two_robots.py get_observation() and YARR/yarr/utils/rollout_generator.py
        """
        observation_data = {}
        perception_data = {}

        wl_image, wl_depth = self.wl_camera.read()
        wr_image, wr_depth = self.wr_camera.read()

        original_height, original_width = wl_image.shape[0], wl_image.shape[1]
        wl_image = self.pad_image(wl_image, mode='constant')
        wl_image = Image.fromarray(wl_image)
        wl_image = wl_image.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
        wl_depth = self.pad_image(wl_depth, mode='constant')[:, :, 0]

        wr_image = self.pad_image(wr_image, mode='constant')
        wr_image = Image.fromarray(wr_image)
        wr_image = wr_image.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
        wr_depth = self.pad_image(wr_depth, mode='constant')[:, :, 0]

        zoom_factor = TARGET_IMAGE_SIZE / wl_depth.shape[0]
        wl_depth = zoom(wl_depth, zoom=zoom_factor, order=1)
        wr_depth = zoom(wr_depth, zoom=zoom_factor, order=1)

        wl_intrinsics = self.adjust_intrinsics_with_padding_and_resize(
                            intrinsics=self.wl_camera_intrinsics,
                            original_width=original_width,
                            original_height=original_height,
                            final_size=TARGET_IMAGE_SIZE,
                        )

        wr_intrinsics = self.adjust_intrinsics_with_padding_and_resize(
                            intrinsics=self.wr_camera_intrinsics,
                            original_width=original_width,
                            original_height=original_height,
                            final_size=TARGET_IMAGE_SIZE,
                        )

        if self.front_cam:
            front_image, front_depth = self.front_camera.read()
            original_height, original_width = front_image.shape[0], front_image.shape[1]
            front_image = self.pad_image(front_image, mode='constant')
            front_image = Image.fromarray(front_image)
            front_image = front_image.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
            front_depth = self.pad_image(front_depth, mode='constant')[:, :, 0]
            front_depth = zoom(front_depth, zoom=zoom_factor, order=1)
            front_intrinsics = self.adjust_intrinsics_with_padding_and_resize(
                                intrinsics=self.front_camera_intrinsics,
                                original_width=original_width,
                                original_height=original_height,
                                final_size=TARGET_IMAGE_SIZE,
                            )

        robot_obs = self.robot.get_observations()

        misc_obj = {
            'wrist_left_camera_extrinsics': np.eye(4),        # not used
            'wrist_left_camera_intrinsics': wl_intrinsics,
            'wrist_left_camera_near': -1,                # not used
            'wrist_left_camera_far': -1,                 # not used
            'wrist_right_camera_extrinsics': np.eye(4,4),       # not used
            'wrist_right_camera_intrinsics': wr_intrinsics,
            'wrist_right_camera_near': -1,               # not used
            'wrist_right_camera_far': -1,                # not used
        }

        if self.front_cam:
            misc_obj['front_camera_extrinsics'] = np.eye(4)        # not used
            misc_obj['front_camera_intrinsics'] = front_intrinsics
            misc_obj['front_camera_near'] = -1                # not used
            misc_obj['front_camera_far'] = -1                 # not used

        wl_gripper_matrix = np.eye(4)
        wl_rotation = R.from_quat(robot_obs['ee_pos_quat'][3:7])
        wl_gripper_matrix[:3,:3] = wl_rotation.as_matrix() 
        wl_gripper_matrix[:3,3] = robot_obs['ee_pos_quat'][:3]

        low_dim_ob_left = UnimanualObservationData(robot_obs['joint_positions'][:7], \
                            robot_obs['joint_positions'][:7], \
                            np.array([]), \
                            robot_obs['gripper_position'][0], \
                            robot_obs['ee_pos_quat'][:7], \
                            wl_gripper_matrix, \
                            np.array([robot_obs['gripper_position'][0], robot_obs['gripper_position'][0]]), \
                            np.array([]), \
                            np.array([1]))

        wr_gripper_matrix = np.eye(4)
        wr_rotation = R.from_quat(robot_obs['ee_pos_quat'][10:])
        wr_gripper_matrix[:3,:3] = wr_rotation.as_matrix() 
        wr_gripper_matrix[:3,3] = robot_obs['ee_pos_quat'][7:10]

        low_dim_ob_right = UnimanualObservationData(robot_obs['joint_positions'][7:], \
                            robot_obs['joint_positions'][7:], \
                            np.array([]), \
                            robot_obs['gripper_position'][1], \
                            robot_obs['ee_pos_quat'][7:], \
                            wr_gripper_matrix, \
                            np.array([robot_obs['gripper_position'][1], robot_obs['gripper_position'][1]]), \
                            np.array([]), \
                            np.array([1]))

        perception_data = {
            'wrist_left_pose': wl_gripper_matrix,
            'wrist_right_pose': wr_gripper_matrix,
            'wrist_left_rgb': np.array(wl_image),
            'wrist_left_depth': np.array(wl_depth),
            'wrist_left_point_cloud': np.zeros((128, 128, 3)),  # not used
            'wrist_right_rgb': np.array(wr_image),
            'wrist_right_depth': np.array(wr_depth),
            'wrist_right_point_cloud': np.zeros((128, 128, 3)), # not used
        }

        if self.front_cam:
            perception_data['front_rgb'] = np.array(front_image)
            perception_data['front_depth'] = np.array(front_depth)
            perception_data['front_point_cloud'] = np.zeros((128, 128, 3)) # not used

        bimanual_ob = BimanualObservation(perception_data, np.array([]), misc_obj)
        bimanual_ob.left = low_dim_ob_left
        bimanual_ob.right = low_dim_ob_right

        obs = self.extract_obs_bimanual(bimanual_ob)
        return obs

    def move_robot(self, target_left_joint_pos, target_right_joint_pos, execute_wo_input):
        current_left_joint_positions = self.robot._robot_l.robot.getj()
        # print('current_left_joint_positions: ', current_left_joint_positions)
        # print('target_left_joint_pos: ', target_left_joint_pos)
        # print('delta left joint positions: ', target_left_joint_pos[:6] - current_left_joint_positions)

        current_right_joint_positions = self.robot._robot_r.robot.getj()
        # print('current_right_joint_positions: ', current_right_joint_positions)
        # print('target_right_joint_pos: ', target_right_joint_pos)
        # print('delta right joint positions: ', target_right_joint_pos[:6] - current_right_joint_positions)

        key = None
        if execute_wo_input:
            key = ''
        else:
            key = input("Press enter to execute or s to skip this action or anything else to exit:\n")

        if key == '':
            p1 = threading.Thread(target=slow_servoj_move, args=(self.robot._robot_l, target_left_joint_pos[:-1], 6))
            p2 = threading.Thread(target=slow_servoj_move, args=(self.robot._robot_r, target_right_joint_pos[:-1], 6))
            # Start processes
            p1.start()
            p2.start()
            # Wait for completion
            p1.join()
            p2.join()

            p1 = threading.Thread(target=execute_gripper_action, args=(self.robot._robot_l, target_left_joint_pos[-1]))
            p2 = threading.Thread(target=execute_gripper_action, args=(self.robot._robot_r, target_right_joint_pos[-1]))
            # Start processes
            p1.start()
            p2.start()
            # Wait for completion
            p1.join()
            p2.join()            

        elif key == 's':
            print(f'Skip this action')
        else:
            gc.collect()
            sys.exit(0)

    def move_robot_to_starting_states(self):
        p1 = threading.Thread(target=slow_servoj_move, args=(self.robot._robot_l, self.left_starting_state))
        p2 = threading.Thread(target=slow_servoj_move, args=(self.robot._robot_r, self.right_starting_state))
        # Start processes
        p1.start()
        p2.start()
        # Wait for completion
        p1.join()
        p2.join()

    def set_env_swift_robot_state(self, left_joint_pos, right_joint_pos):
        def slow_servoj_env_swift_move(robot, q_target, steps=50):
            q_current = robot.q
            for i in range(steps):
                    q_interp = [(1 - i/steps) * q_current[j] + (i/steps) * q_target[j] for j in range(6)]
                    robot.q = q_interp
                    self.env_swift.step(self.env_swift_dt)
        key = input("Press enter to execute the action in swift or anything else to exit:\n")
        if key == '':
            p1 = threading.Thread(target=slow_servoj_env_swift_move, args=(self.env_swift_left_robot, left_joint_pos))
            p2 = threading.Thread(target=slow_servoj_env_swift_move, args=(self.env_swift_right_robot, right_joint_pos))
            # Start processes
            p1.start()
            p2.start()
            # Wait for completion
            p1.join()
            p2.join()       
        else:
            print('Exit')
            gc.collect()
            sys.exit(0)

def eval_seed(train_cfg, eval_cfg, logdir, env_device, multi_task, seed, env_config) -> None:
    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    train_cfg.method.robot_name = eval_cfg.method.robot_name

    agent = agent_factory.create_agent(train_cfg, eval_cfg=eval_cfg)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(logdir, 'weights')

    # evaluate a specific checkpoint
    if type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)
    else:
        raise Exception("Unknown eval type")

    # csv file
    csv_file_exist = False
    csv_filename = 'eval.csv'

    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info("No weights to evaluate. Results are already available in eval_data.csv")
        sys.exit(0)

    if env_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    env = RobotEnv(train_cfg, eval_cfg, env_config)

    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    # evaluate a single checkpoint 
    # here, we assume it's going to be robot execution
    # real-world robot execution
    env.initialize_robot('192.10.0.12', '192.10.0.11')
    key = input("Press enter to move robots to starting states or anything else to exit:\n")
    if key == '':
        env.move_robot_to_starting_states()
    else:
        print('Exit')
        gc.collect()
        sys.exit(0)
    key = input("Press enter to get the observation else to exit:\n")
    if key == '':
        pass      
    else:
        print('Exit')
        gc.collect()
        sys.exit(0)

    # initialize policy
    agent.build(training=False, device=device)
    weight_path = os.path.join(weightsdir, str(weight_folders[0]))
    agent.load_weights(weight_path)

    env._i = 0
    agent.reset()
    step_signal = Value('i', -1)
    obs = env.get_observation()
    obs_history = {k: [np.array(v, dtype=env._get_type(v))] * env._timesteps for k, v in obs.items()}
    prev_target_right_joint_pos = None
    prev_target_left_joint_pos = None
    for step in range(eval_cfg.rlbench.episode_length):
        if step % env.real_world_execute_k_actions == 0:
            execute_wo_input = False
        else:
            execute_wo_input = True
        prepped_data = {k:torch.tensor(np.array(v)[None], device=device) for k, v in obs_history.items()}

        # for debugging rgb images...
        # import matplotlib.pyplot as plt
        # # Remove extra dimensions: [1, 1, 3, 128, 128] → [3, 128, 128]
        # img_tensor = prepped_data['wrist_left_rgb'].squeeze(0).squeeze(0)
        # # Convert to NumPy and transpose: [3, 128, 128] → [128, 128, 3]
        # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        # # Normalize to [0, 1] if necessary (e.g., if image values are in [-1, 1] or other range)
        # img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        # # Visualize
        # plt.imshow(img_np)
        # plt.axis('off')
        # plt.show()

        act_result = agent.act(step_signal.value, prepped_data,
                                deterministic=eval)
        target_right_joint_pos = act_result.action[:7]
        target_left_joint_pos = act_result.action[8:15]

        print('target_right_joint_pos: ', target_right_joint_pos)
        print('target_left_joint_pos: ', target_left_joint_pos)

        if prev_target_right_joint_pos is None or prev_target_left_joint_pos is None:
            prev_target_right_joint_pos = target_right_joint_pos
            prev_target_left_joint_pos = target_left_joint_pos
        else:
            print('Delta right joint positions: ', target_right_joint_pos - prev_target_right_joint_pos)
            print('Delta left joint positions: ', target_left_joint_pos - prev_target_left_joint_pos)
            prev_target_right_joint_pos = target_right_joint_pos
            prev_target_left_joint_pos = target_left_joint_pos
        
        if env.real_world_viz and not execute_wo_input:
            env.set_env_swift_robot_state(target_left_joint_pos, target_right_joint_pos)

        env.move_robot(target_left_joint_pos, target_right_joint_pos, execute_wo_input)
        env._i += 1
        obs = env.get_observation()
        obs_history = {k: [np.array(v, dtype=env._get_type(v))] * env._timesteps for k, v in obs.items()}
    del env
    del agent
    gc.collect()
    torch.cuda.empty_cache()  


@hydra.main(config_name="eval", config_path="conf", version_base="1.1")
def main(eval_cfg: DictConfig) -> None:
    logging.info("\n" + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(
        eval_cfg.framework.logdir,
        eval_cfg.rlbench.task_name,
        eval_cfg.method.name,
        "seed%d" % start_seed,
    )

    train_config_path = os.path.join(logdir, "config.yaml")

    if os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception(f"Missing seed{start_seed}/config.yaml. Logdir is {logdir}")

    # sanity checks
    assert train_cfg.method.name == eval_cfg.method.name
    assert train_cfg.method.agent_type == eval_cfg.method.agent_type
    for task in eval_cfg.rlbench.tasks:
        assert task in train_cfg.rlbench.tasks

    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info("Using env device %s." % str(env_device))

    gripper_mode = eval(eval_cfg.rlbench.gripper_mode)()
    arm_action_mode = eval(eval_cfg.rlbench.arm_action_mode)()
    action_mode = eval(eval_cfg.rlbench.action_mode)(arm_action_mode, gripper_mode)

    is_bimanual = eval_cfg.method.robot_name == "bimanual"

    if is_bimanual:
        task_path = rlbench_task.BIMANUAL_TASKS_PATH
    else:
        task_path = rlbench_task.TASKS_PATH

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(task_path)
        if t != "__init__.py" and t.endswith(".py")
    ]
    eval_cfg.rlbench.cameras = (
        eval_cfg.rlbench.cameras
        if isinstance(eval_cfg.rlbench.cameras, ListConfig)
        else [eval_cfg.rlbench.cameras]
    )
    obs_config = observation_utils.create_obs_config(
        eval_cfg.rlbench.cameras,
        eval_cfg.rlbench.camera_resolution,
        eval_cfg.method.name,
        eval_cfg.method.robot_name,
    )

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    multi_task = len(eval_cfg.rlbench.tasks) > 1

    tasks = eval_cfg.rlbench.tasks
    task_classes = [None]
    # for task in tasks:
    #     if task not in task_files:
    #         raise ValueError("Task %s not recognised!." % task)
    #     task_classes.append(task_file_to_task_class(task, is_bimanual))

    # single-task or multi-task
    if multi_task:
        env_config = (
            task_classes,
            obs_config,
            action_mode,
            eval_cfg.rlbench.demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            eval_cfg.framework.eval_episodes,
            train_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )
    else:
        env_config = (
            task_classes[0],
            obs_config,
            action_mode,
            eval_cfg.rlbench.demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            train_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )

    logging.info("Evaluating seed %d." % start_seed)
    eval_seed(
        train_cfg,
        eval_cfg,
        logdir,
        env_device,
        multi_task,
        start_seed,
        env_config,
    )


if __name__ == "__main__":
    peract_config.on_init()
    main()
