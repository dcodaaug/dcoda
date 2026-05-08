# 1
CUDA_DEVICE=1
# 2
TASK_NAME="coordinated_lift_ball"
METHOD="ACT_BC_LANG"
IMAGE_SIZE=128
DEMO_PATH="/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test"
# 3
EXP_NAME="/nas/datasets/zsh/MVDA_ckpts/ACT/logs/2026_04_13_03_01_coordinated_lift_ball_real+daug_300_43"
# 4
EVAL_TYPE=[214000,224000,234000,244000,254000,212000,222000,232000,242000,252000]
SEED=43

DISPLAY=:99 CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval.py \
    method=${METHOD} \
    rlbench.task_name=${EXP_NAME} \
    rlbench.tasks=[${TASK_NAME}] \
    rlbench.demo_path=${DEMO_PATH} \
    rlbench.camera_resolution=[${IMAGE_SIZE},${IMAGE_SIZE}] \
    rlbench.cameras=[wrist_right,wrist_left] \
    rlbench.episode_length=400 \
    rlbench.gripper_mode=BimanualGripperJointPosition \
    rlbench.arm_action_mode=BimanualJointPosition \
    rlbench.action_mode=BimanualJointPositionActionMode \
    framework.logdir=/home/zsh/dcoda/11/logs \
    framework.eval_episodes=25 \
    framework.eval_type=${EVAL_TYPE} \
    framework.start_seed=${SEED} \
    framework.eval_envs=10





# 1
CUDA_DEVICE=1
# 2
TASK_NAME="coordinated_lift_ball"
METHOD="ACT_BC_LANG"
IMAGE_SIZE=128
DEMO_PATH="/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test"
# 3
EXP_NAME="/nas/datasets/zsh/MVDA_ckpts/ACT/logs/2026_04_01_03_49_coordinated_lift_ball_real+daug_125_no_smooth_43"
# 4
EVAL_TYPE=[214000,224000,234000,244000,254000,212000,222000,232000,242000,252000]
SEED=43

DISPLAY=:99 CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval.py \
    method=${METHOD} \
    rlbench.task_name=${EXP_NAME} \
    rlbench.tasks=[${TASK_NAME}] \
    rlbench.demo_path=${DEMO_PATH} \
    rlbench.camera_resolution=[${IMAGE_SIZE},${IMAGE_SIZE}] \
    rlbench.cameras=[wrist_right,wrist_left] \
    rlbench.episode_length=400 \
    rlbench.gripper_mode=BimanualGripperJointPosition \
    rlbench.arm_action_mode=BimanualJointPosition \
    rlbench.action_mode=BimanualJointPositionActionMode \
    framework.logdir=/home/zsh/dcoda/11/logs \
    framework.eval_episodes=25 \
    framework.eval_type=${EVAL_TYPE} \
    framework.start_seed=${SEED} \
    framework.eval_envs=10




# 1
CUDA_DEVICE=1
# 2
TASK_NAME="coordinated_lift_ball"
METHOD="ACT_BC_LANG"
IMAGE_SIZE=128
DEMO_PATH="/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test"
# 3
EXP_NAME="/nas/datasets/zsh/MVDA_ckpts/ACT/logs/2026_03_31_04_48_coordinated_lift_ball_real+daug_150_demos_no_smooth_43"
# 4
EVAL_TYPE=[214000,224000,234000,244000,254000,212000,222000,232000,242000,252000]
SEED=43

DISPLAY=:99 CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval.py \
    method=${METHOD} \
    rlbench.task_name=${EXP_NAME} \
    rlbench.tasks=[${TASK_NAME}] \
    rlbench.demo_path=${DEMO_PATH} \
    rlbench.camera_resolution=[${IMAGE_SIZE},${IMAGE_SIZE}] \
    rlbench.cameras=[wrist_right,wrist_left] \
    rlbench.episode_length=400 \
    rlbench.gripper_mode=BimanualGripperJointPosition \
    rlbench.arm_action_mode=BimanualJointPosition \
    rlbench.action_mode=BimanualJointPositionActionMode \
    framework.logdir=/home/zsh/dcoda/11/logs \
    framework.eval_episodes=25 \
    framework.eval_type=${EVAL_TYPE} \
    framework.start_seed=${SEED} \
    framework.eval_envs=10



# 1
CUDA_DEVICE=1
# 2
TASK_NAME="coordinated_lift_ball"
METHOD="ACT_BC_LANG"
IMAGE_SIZE=128
DEMO_PATH="/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test"
# 3
EXP_NAME="/nas/datasets/zsh/MVDA_ckpts/ACT/logs/2026_04_13_02_49_coordinated_lift_ball_real+dcoda_300_43"
# 4
EVAL_TYPE=[214000,224000,234000,244000,254000]
SEED=43

DISPLAY=:99 CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval.py \
    method=${METHOD} \
    rlbench.task_name=${EXP_NAME} \
    rlbench.tasks=[${TASK_NAME}] \
    rlbench.demo_path=${DEMO_PATH} \
    rlbench.camera_resolution=[${IMAGE_SIZE},${IMAGE_SIZE}] \
    rlbench.cameras=[wrist_right,wrist_left] \
    rlbench.episode_length=400 \
    rlbench.gripper_mode=BimanualGripperJointPosition \
    rlbench.arm_action_mode=BimanualJointPosition \
    rlbench.action_mode=BimanualJointPositionActionMode \
    framework.logdir=/home/zsh/dcoda/11/logs \
    framework.eval_episodes=25 \
    framework.eval_type=${EVAL_TYPE} \
    framework.start_seed=${SEED} \
    framework.eval_envs=5





# 1
CUDA_DEVICE=1
# 2
TASK_NAME="coordinated_lift_ball"
METHOD="ACT_BC_LANG"
IMAGE_SIZE=128
DEMO_PATH="/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test"
# 3
EXP_NAME="/nas/datasets/zsh/MVDA_ckpts/ACT/logs/2026_04_01_03_49_coordinated_lift_ball_real+dcoda_125_43"
# 4
EVAL_TYPE=[214000,224000,234000,244000,254000]
SEED=43

DISPLAY=:99 CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval.py \
    method=${METHOD} \
    rlbench.task_name=${EXP_NAME} \
    rlbench.tasks=[${TASK_NAME}] \
    rlbench.demo_path=${DEMO_PATH} \
    rlbench.camera_resolution=[${IMAGE_SIZE},${IMAGE_SIZE}] \
    rlbench.cameras=[wrist_right,wrist_left] \
    rlbench.episode_length=400 \
    rlbench.gripper_mode=BimanualGripperJointPosition \
    rlbench.arm_action_mode=BimanualJointPosition \
    rlbench.action_mode=BimanualJointPositionActionMode \
    framework.logdir=/home/zsh/dcoda/11/logs \
    framework.eval_episodes=25 \
    framework.eval_type=${EVAL_TYPE} \
    framework.start_seed=${SEED} \
    framework.eval_envs=5