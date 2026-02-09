import os 
import json
import numpy as np
import collections

def read_pose(file):
    data = json.load(open(file, "r"))
    dct = collections.defaultdict(list)
    frame_names = sorted(list(data.keys()))
    for frame in frame_names:
        pose = data[frame]
        pose = np.asarray(pose)
        orb_pose = orb_to_blender(pose)
        dct['imgs'].append("images/" + frame)
        dct['poses'].append(orb_pose)
        dct['poses_orig'].append(pose)
    return dict(dct)

def read_bimanual_pose(file):
    data = json.load(open(file, "r"))
    dct_wrist_left = collections.defaultdict(list)
    dct_wrist_right = collections.defaultdict(list)
    frame_names = sorted(list(data.keys()))
    for frame in frame_names:
        pose = data[frame]
        pose = np.asarray(pose)
        orb_pose = orb_to_blender(pose)

        if 'wrist_left' in frame:
            dct_wrist_left['imgs'].append("images/" + frame)
            dct_wrist_left['poses'].append(orb_pose)
            dct_wrist_left['poses_orig'].append(pose)
        elif 'wrist_right' in frame:
            dct_wrist_right['imgs'].append("images/" + frame)
            dct_wrist_right['poses'].append(orb_pose)
            dct_wrist_right['poses_orig'].append(pose)
        else:
            raise NotImplementedError

    return dict(dct_wrist_left), dict(dct_wrist_right)

def read_gripper_state_bimanual(file):
    data = json.load(open(file, "r"))
    dct_gripper = collections.defaultdict(list)
    frame_names = sorted(list(data.keys()))
    for frame in frame_names:
        gripper_state = data[frame]
        gripper_state = np.asarray(gripper_state)
        dct_gripper[frame] = gripper_state

    return dict(dct_gripper)

def read_gripper_matrices_bimanual(file):
    data = json.load(open(file, "r"))
    left_dct_gripper_matrices = collections.defaultdict(list)
    right_dct_gripper_matrices = collections.defaultdict(list)
    frame_names = sorted(list(data.keys()))
    for frame in frame_names:
        gripper_matrix = data[frame]
        gripper_matrix = np.asarray(gripper_matrix)

        if 'wrist_left' in frame:
            left_dct_gripper_matrices['gripper'].append(gripper_matrix)
        elif 'wrist_right' in frame:
            right_dct_gripper_matrices['gripper'].append(gripper_matrix)
        else:
            raise NotImplementedError
    return dict(left_dct_gripper_matrices), dict(right_dct_gripper_matrices)

def orb_to_blender(camera_local):
    pre_conversion = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1],
    ])
    conversion = np.array([
            [1,0,0,0],
            [0,0,1,0],
            [0,-1,0,0],
            [0,0,0,1],
    ])
    
    orb_world = np.matmul(camera_local,pre_conversion)
    blender_world = np.matmul(conversion,orb_world)

    return blender_world