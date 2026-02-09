import os
import sys
import argparse
import pathlib
import json
import numpy as np
from scipy.spatial.transform import Rotation as scir
import torch

script_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(os.path.join(src_path, "datasets"))
from write_translations import get_colmap_labels
from slam_utils import read_pose, read_bimanual_pose, read_gripper_state_bimanual

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

            if args.sfm_method == "orbslam_bimanual":
                dct_left["focal_y"] = float(focal_length/img_h)
                dct_right["focal_y"] = float(focal_length/img_h)
                data[folder_path] = {'dct_left': dct_left, 'dct_right': dct_right, 'dct_gripper': dct_gripper}
            else:
                dct["focal_y"] = float(focal_length/img_h)
                data[folder_path] = dct
    
    pre_conversion = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # Here we assume all of the tasks use the same range for x, y, z rotations and magnitudes
    x_euler_range = [-0.5,0.5]
    y_euler_range = [-0.5,0.5]
    z_euler_range = [-0.5,0.5]
    magnitude_l, magnitude_u = 0.01,0.02
    
    generation_data = []
    folders = sorted(list(data.keys()))
    image_index = 0
    for folder in folders:
        if args.sfm_method == "orbslam_bimanual":
            left_folder_data = data[folder]['dct_left']
            right_folder_data = data[folder]['dct_right']
            gripper_data = data[folder]['dct_gripper']
            num_imgs = len(left_folder_data['imgs'])
        else:
            folder_data = data[folder]
            num_imgs = len(folder_data['imgs'])
        
        gripper_state_file = os.path.join(folder, "gripper_state.json")
        if os.path.exists(gripper_state_file):
            gripper_state = json.load(open(gripper_state_file, 'r'))
        else:
            gripper_state = None
        
        for frame_idx in range(0, num_imgs, args.every_x_frame):
            if args.sfm_method == "orbslam_bimanual":
                left_full_img_path = os.path.join(folder, left_folder_data['imgs'][frame_idx])
                right_full_img_path = os.path.join(folder, right_folder_data['imgs'][frame_idx])
                if gripper_data is not None:
                    gripper_key = left_folder_data['imgs'][frame_idx].split('_')[-1].split('.')[0]
            else:
                full_img_path = os.path.join(folder, folder_data['imgs'][frame_idx])
            
            if args.sfm_method == "orbslam_bimanual":
                # (original implementation) Skip frames where the grabber is opening/closing
                if gripper_state is not None:
                    raise NotImplementedError
                left_orig_img_copy_path = os.path.join(output_root, 'images', "left_%09d_o.png" % image_index)
                right_orig_img_copy_path = os.path.join(output_root, 'images', "right_%09d_o.png" % image_index)
            else:
                # (original implementation) Skip frames where the grabber is opening/closing
                if gripper_state is not None:
                    img_key = full_img_path.split("/")[-1]
                    if img_key in gripper_state:
                        if gripper_state[img_key] < 0:
                            continue 
                orig_img_copy_path = os.path.join(output_root, 'images', "%09d_o.png" % image_index)

            if args.sfm_method == "orbslam_bimanual":
                # bimanual implementation
                for i in range (args.mult):
                    left_output_img_path = os.path.join(output_root, "images", "left_%09d.png" % image_index)
                    right_output_img_path = os.path.join(output_root, "images", "right_%09d.png" % image_index)
                    
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
                    
                    # assume same focal_y for both cameras
                    this_data = {
                        "left_img": left_full_img_path,
                        "right_img": right_full_img_path,
                        "left_orig_img_copy_path" : left_orig_img_copy_path,
                        "right_orig_img_copy_path" : right_orig_img_copy_path,
                        "focal_y": left_folder_data["focal_y"],
                        "left_output": left_output_img_path,
                        "right_output": right_output_img_path,
                        "magnitude_left" : magnitude_left,
                        "transformation_left": transformation_left.tolist(),
                        "magnitude_right" : magnitude_right,
                        "transformation_right": transformation_right.tolist(),
                    }
                    if args.sample_rotation:
                        this_data["rot_euler_left"] = rot_euler_left.astype(float).tolist()
                        this_data["rot_euler_right"] = rot_euler_right.astype(float).tolist()
                    if gripper_data is not None:
                        this_data["gripper_data"] = gripper_data[gripper_key].tolist()
                    generation_data.append(this_data)
                    image_index += 1
            else:
                # original implementation
                for i in range (args.mult):
                    output_img_path = os.path.join(output_root, "images", "%09d.png" % image_index)
                    
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    # Sample a random magnitude between 0 and 5
                    magnitude = np.random.uniform(magnitude_l,magnitude_u)
                    # Create a translation vector with the random direction and magnitude
                    translation = direction * magnitude
                    transformation = np.eye(4)
                    transformation[:3, 3] = translation
                    
                    if args.sample_rotation:
                        if x_euler_range is not None:
                            x_euler = np.random.uniform(*x_euler_range)
                        else:
                            x_euler = 0
                        if y_euler_range is not None:
                            
                            y_euler = np.random.uniform(*y_euler_range)
                        else:
                            y_euler = 0
                        if z_euler_range is not None:
                            z_euler = np.random.uniform(*z_euler_range)
                        else:
                            z_euler = 0
                        rot_euler = np.array([x_euler, y_euler, z_euler])
                        rot = scir.from_euler('xyz', rot_euler, degrees=True).as_matrix()
                        transformation[:3,:3] = rot                
                    transformation = np.matmul(transformation, pre_conversion)
                    transformation = np.matmul(np.linalg.inv(pre_conversion),transformation)
                    
                    this_data = {
                        "img": full_img_path,
                        "orig_img_copy_path" : orig_img_copy_path,
                        "focal_y": folder_data["focal_y"],
                        "output": output_img_path,
                        "magnitude" : magnitude,
                        "transformation": transformation.tolist()
                    }
                    if args.sample_rotation:
                        this_data["rot_euler"] = rot_euler.astype(float).tolist()
                    generation_data.append(this_data)
                    image_index += 1
    
    print("Total number of images:", len(generation_data))
    out_file = os.path.join(output_root, 'data.json')
    if os.path.exists(out_file):
        print(f"{out_file} file exists")
    else:
        with open(out_file, 'w') as f: json.dump(generation_data, f)
        print("Written to: ", out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task',default='push',type=str)
    parser.add_argument('--sfm_method', type=str, choices=['colmap', 'grabber_orbslam', 'orbslam_bimanual'])
    parser.add_argument("--data_folders", type=str, nargs="+", default=[])
    parser.add_argument("--focal_lengths", type=float, nargs="+", default=[])
    parser.add_argument("--image_heights", type=float, nargs="+", default=[])
    parser.add_argument('--output_root', type=str, help="output folder")
    parser.add_argument('--suffix', default=None, help="suffix to add to output_root, if you want to generate multiple versions")
    parser.add_argument('--sample_rotation',action='store_true')
    parser.add_argument('--every_x_frame', type=int, default=10, help="Generate augmenting images of every every_x_frame-th frame. If the demonstrations are recorded at high frame-rate (e.g. above 5fps), nearby frames are very similar, and there are too many frames in each trajectory, so it is not necessary to generate augmenting samples for every frame.")
    parser.add_argument('--mult',type=int,default=3,help='Number of augmenting images to generate per input frame')
    parser.add_argument('--seed', type=int, default=0, help="Random seed number.")

    args = parser.parse_args()

    main(args)