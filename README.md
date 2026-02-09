# D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation

[[Project website](https://dcodaaug.github.io/D-CODA/)] [[Paper](https://arxiv.org/abs/2505.04860)]

This project is a PyTorch implementation of D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation, published in CoRL 2025.

**Authors**: [I-Chun Arthur Liu](https://arthurliu.com/), [Jason Chen](https://jasoonchen.com/), [Gaurav S. Sukhatme](https://uscresl.org/principal-investigator/), [Daniel Seita](https://danielseita.github.io/).

Learning bimanual manipulation is challenging due to its high dimensionality and tight coordination required between two arms. Eye-in-hand imitation learning, which uses wrist-mounted cameras, simplifies perception by focusing on task-relevant views. However, collecting diverse demonstrations remains costly, motivating the need for scalable data augmentation. While prior work has explored visual augmentation in single-arm settings, extending these approaches to bimanual manipulation requires generating viewpoint-consistent observations across both arms and producing corresponding action labels that are both valid and feasible. In this work, we propose Diffusion for COordinated Dual-arm Data Augmentation (D-CODA), a method for offline data augmentation tailored to eye-in-hand bimanual imitation learning that trains a diffusion model to synthesize novel, viewpoint-consistent wrist-camera images for both arms while simultaneously generating joint-space action labels. It employs constrained optimization to ensure that augmented states involving gripper-to-object contacts adhere to constraints suitable for bimanual coordination. We evaluate D-CODA on 5 simulated and 3 real-world tasks. Our results across 2250 simulation trials and 180 real-world trials demonstrate that it outperforms baselines and ablations, showing its potential for scalable data augmentation in eye-in-hand bimanual manipulation.

## Installation

### Prerequisites

The codebase is built on [PerAct^2](https://github.com/markusgrotz/peract_bimanual), which is based on [PerAct](https://peract.github.io), which in turn is built on the [ARM repository](https://github.com/stepjam/ARM) by James et al. The prerequisites are the same as for PerAct or ARM.

#### 1. Environment


Install miniconda if not already present on the current system.
You can use `scripts/install_conda.sh` for this step:

```bash
sudo apt install curl 

curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh
```

Next, create the rlbench environment and install the dependencies

```bash
conda create -n rlbench python=3.8
conda activate rlbench
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```


#### 2. Dependencies

You need to setup RBench, PyRep, and YARR. 

**Install PyRep:**

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)
- [Ubuntu 22.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PERACT_BIMANUAL_ROOT=<EDIT ME>/PATH/TO/PERACT_BIMANUAL_DIR
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
cd PyRep
pip install -e .
```

**Install RLBench:**

```bash
cd ../RLBench
pip install -e .
```

**Install YARR:**

```bash
cd ../YARR
pip install -e .
```

**Install PyTorch3D:**

Make sure you have [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) installed.

```bash
cd ../
mkdir logs # create logs folder for storing experimental files/checkpoints
conda install -n rlbench gxx_linux-64
git clone https://github.com/facebookresearch/pytorch3d.git pytorch3d
cd pytorch3d
conda run -n rlbench pip install . # installing pytorch3d might take a while
```

**[Optional] Real Robot:**
```bash
pip3 install roboticstoolbox-python
# this handles the asyncio "no running event loop" bug
pip install websockets==13.1
```


**Download Pre-trained Weights for Model Training**

Create data folder:
```bash
cd ../ # make sure you're at the top level
mkdir data
cd data
```

Download this [zip file](https://huggingface.co/datasets/arthur801031/voxact-b/blob/main/pretrained_weights.zip), unzip it, and place `clip_rn50.pth` inside the `data` folder.


#### 3. Install Diffusion Meets Dagger (DMD)

```bash
cd ../DMD

conda env create -f environment.yaml # create a new conda environment
```

Please refer to [DMD/README.md](DMD/README.md) for DMD-specific details.


#### 4. Running `generate_inference_json_vlms.py`
Place [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) into `dcoda/data`.

```bash
# Create a new conda environment
conda create -n sam2 python=3.10

# Outside of dcoda
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

# cd to dcoda
cd PyRep
pip install -e .

cd ../RLBench
pip install -e .

cd ../YARR
pip install -e .

pip uninstall opencv-python
pip install opencv-python-headless

pip3 install roboticstoolbox-python
# this handles the asyncio "no running event loop" bug
pip install websockets==13.1
```

## Training

1. Generate RLBench training data
```bash
cd RLBench/tools
python dataset_generator_bimanual.py
```

2. Convert RLBench training data into D-CODA format
```bash
# make sure you're at dcoda/ before executing the following command

python scripts/convert_dcoda_format_bimanual.py \
--input_dir=./data/rlbench_data/train/coordinated_lift_ball_100_demos_128x128/coordinated_lift_ball/all_variations/episodes \
--output_dir=./DMD/instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual  \
--save_relative_poses \
--save_absolute_poses \
--save_depth_images \
--intervals=6
```

3. Train diffusion model
```bash
# make sure you're at dcoda/DMD/src/ before executing the following command

python -m torch.distributed.launch --use_env --master_port=25678 scripts/mini_train.py \
--batch_size 4 \
--task coordinated_lift_ball \
--sfm_method orbslam_bimanual \
--num_gpus_per_node 1 \
--instance_data_path ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1 \
--colmap_data_folders ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
--focal_lengths -110.85124795436963 \
--max_epoch 20000 \
--checkpoint_retention_interval 500 \
--wandb
```

4. Use trained diffusion model to synthesize images
```bash
# make sure you're at dcoda/DMD/src/ before executing the following commands

python scripts/generate_inference_json_vlms.py \
--task coordinated_lift_ball --sfm_method orbslam_bimanual \
--data_folders ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
--focal_lengths -110.85124795436963 --image_heights 128 \
--suffix v1 \
--output_root ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual \
--sample_rotation --every_x_frame 6 --mult 1 --seed 10 --save_robot_traj_visualization --save_print_to_file


python scripts/sample-imgs-multi.py ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1/data.json \
--model ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/checkpoints/17000-00000000.pth \
--sfm_method orbslam_bimanual \
--image_size 128 --batch_size 16 --gpus 0,1
```

5. Generate corresponding action labels
```bash
python scripts/annotate.py --sfm_method orbslam_bimanual \
--inf_json ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1/data.json \
--gt_json labels_10_bimanual.json --task_data_json labels_10_bimanual.json \
--output_dir ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1_action_labels \
--task_data_dir ../instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual
```

6. Generate the augmented dataset based on the original dataset
- Before you execute this command, create a new folder (e.g. coordinated_lift_ball_200_demos_128x128) in `dcoda/data/rlbench_data/train` that contains the original demonstrations that you want to apply the data augmentations to.
```bash
# make sure you're at dcoda/ before executing the following command

python scripts/convert_to_rlbench_data_bimanual_v4.py \
--org_dir=./data/rlbench_data/train/coordinated_lift_ball_200_demos_128x128/coordinated_lift_ball/all_variations/episodes \
--action_labels_dir=./DMD/instance-data/250125_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1_action_labels 
```
7. Train ACT on the new dataset (original + augmented data)
```bash
# make sure you're at dcoda/ before executing the following command
./scripts_train_eval/train_act.sh
```

## Evaluation

Validation:

```bash
bash ./scripts_train_eval/pc_act_eval_v2.sh \
0 \
coordinated_lift_ball \
data/rlbench_data/val/coordinated_lift_ball_25_demos_128x128 \
2025_04_21_23_21_coordinated_lift_ball_200_new_sampling_v2_run1_0 \
missing \
0
```

Test:
```bash
bash ./scripts_train_eval/pc_act_eval_v2_videos.sh \
0 \
coordinated_lift_ball \
data/rlbench_data/test/coordinated_lift_ball_25_demos_128x128 \
2025_04_21_23_21_coordinated_lift_ball_200_new_sampling_v2_run1_0 \
134000 \
0
```




## Acknowledgements

This repository uses code from the following open-source projects:

#### ARM 
Original:  [https://github.com/stepjam/ARM](https://github.com/stepjam/ARM)  
License: [ARM License](https://github.com/stepjam/ARM/LICENSE)    
Changes: Data loading was modified for PerAct. Voxelization code was modified for DDP training.

#### PerceiverIO
Original: [https://github.com/lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)   
License: [MIT](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)   
Changes: PerceiverIO adapted for 6-DoF manipulation.

#### ViT
Original: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)     
License: [MIT](https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE)   
Changes: ViT adapted for baseline.   

#### LAMB Optimizer
Original: [https://github.com/cybertronai/pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)   
License: [MIT](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)   
Changes: None.

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to extract token and sentence features.

Thanks for open-sourcing! 

## Licenses
- [PerAct License (Apache 2.0)](LICENSE) - Perceiver-Actor Transformer
- [ARM License](ARM_LICENSE) - Voxelization and Data Preprocessing 
- [YARR Licence (Apache 2.0)](https://github.com/stepjam/YARR/blob/main/LICENSE)
- [RLBench Licence](https://github.com/stepjam/RLBench/blob/master/LICENSE)
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE)
- [Perceiver PyTorch License (MIT)](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)
- [LAMB License (MIT)](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)
- [CLIP License (MIT)](https://github.com/openai/CLIP/blob/main/LICENSE)

## Release Notes

**Update 2024-07-10**

Initial release


## Citations 


**PerAct^2**
```
@misc{grotz2024peract2,
      title={PerAct2: Benchmarking and Learning for Robotic Bimanual Manipulation Tasks}, 
      author={Markus Grotz and Mohit Shridhar and Tamim Asfour and Dieter Fox},
      year={2024},
      eprint={2407.00278},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.00278}, 
}
```

**PerAct**
```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

**C2FARM**
```
@inproceedings{james2022coarse,
  title={Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation},
  author={James, Stephen and Wada, Kentaro and Laidlow, Tristan and Davison, Andrew J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13739--13748},
  year={2022}
}
```

**PerceiverIO**
```
@article{jaegle2021perceiver,
  title={Perceiver io: A general architecture for structured inputs \& outputs},
  author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
  journal={arXiv preprint arXiv:2107.14795},
  year={2021}
}
```

**RLBench**
```
@article{james2020rlbench,
  title={Rlbench: The robot learning benchmark \& learning environment},
  author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3019--3026},
  year={2020},
  publisher={IEEE}
}
```

## Questions or Issues?

Please file an issue with the issue tracker.  
