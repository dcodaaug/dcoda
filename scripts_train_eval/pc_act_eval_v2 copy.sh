#!/usr/bin/env bash
set -euo pipefail

# Use one or more GPU ids, e.g. (0 1). Jobs are round-robin assigned.
CUDA_DEVICES=(1)
PARALLEL_WORKERS=4

TASK_NAME="coordinated_lift_ball"
METHOD="ACT_BC_LANG"
IMAGE_SIZE=128
DEMO_PATH="/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test"
EXP_NAME="/nas/datasets/zsh/MVDA_ckpts/ACT/logs/2026_03_18_03_25_coordinated_lift_ball_real+daug_200_demos_43"
EVAL_TYPE=260000
SEED=43
TOTAL_EVAL_EPISODES=25

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

run_eval_shard() {
    local gpu_id="$1"
    local start_ep="$2"
    local num_eps="$3"
    local log_file="$4"

    DISPLAY=:99 CUDA_VISIBLE_DEVICES="${gpu_id}" python eval.py \
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
        framework.eval_episodes=${num_eps} \
        framework.eval_from_eps_number=${start_ep} \
        framework.eval_save_metrics=False \
        framework.eval_type=${EVAL_TYPE} \
        framework.start_seed=${SEED} \
        > "${log_file}" 2>&1
}

if (( PARALLEL_WORKERS <= 1 )); then
    run_eval_shard "${CUDA_DEVICES[0]}" 0 "${TOTAL_EVAL_EPISODES}" "/dev/stdout"
    exit 0
fi

WORKERS=${PARALLEL_WORKERS}
if (( WORKERS > TOTAL_EVAL_EPISODES )); then
    WORKERS=${TOTAL_EVAL_EPISODES}
fi

TMP_LOG_DIR="/tmp/pc_act_eval_v2_${SEED}_$(date +%s)"
mkdir -p "${TMP_LOG_DIR}"

declare -a PIDS
declare -a LOG_FILES

for (( i=0; i<WORKERS; i++ )); do
    start_ep=$(( i * TOTAL_EVAL_EPISODES / WORKERS ))
    end_ep=$(( (i + 1) * TOTAL_EVAL_EPISODES / WORKERS ))
    num_eps=$(( end_ep - start_ep ))
    gpu_id="${CUDA_DEVICES[$(( i % ${#CUDA_DEVICES[@]} ))]}"
    log_file="${TMP_LOG_DIR}/shard_${i}.log"

    run_eval_shard "${gpu_id}" "${start_ep}" "${num_eps}" "${log_file}" &
    PIDS+=("$!")
    LOG_FILES+=("${log_file}")
    echo "Launched shard ${i}: gpu=${gpu_id}, episodes=${start_ep}..$((end_ep - 1))"
done

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

echo "---- Per-shard summary ----"
for f in "${LOG_FILES[@]}"; do
    tail -n 3 "${f}" || true
done

echo "---- Aggregated episode score ----"
awk -F'Score: ' '
    /\| Episode [0-9]+ \| Score:/ {
        split($2, parts, " \\| ")
        sum += parts[1]
        count += 1
    }
    END {
        if (count == 0) {
            print "No episode score lines parsed."
            exit 1
        }
        printf("Episodes parsed: %d\n", count)
        printf("Mean score: %.6f\n", sum / count)
    }
' "${LOG_FILES[@]}"

if (( failed != 0 )); then
    echo "One or more shards failed. Full logs: ${TMP_LOG_DIR}"
    exit 1
fi

echo "Parallel evaluation completed. Logs: ${TMP_LOG_DIR}"