#!/bin/bash

# experiment_queue.sh - 简洁的实验队列

echo "开始执行实验队列..."

# ==================== 实验命令列表 ====================
# 在这里添加你的bash命令，每行一个

BASH_PATH="/share/yangxizhong/workspace/step_level_deepconf/scripts/run_deepconf_snapshot.sh"

SCRIPT_PATH="/share/yangxizhong/workspace/step_level_deepconf/scripts/example_offline.py"
MODEL_DIR="/share/yangxizhong/workspace/step_level_deepconf/model"
MODEL_PATH="/share/yangxizhong/ckpt/DeepSeek-R1-0528-Qwen3-8B"
BUDGET=64
TENSOR_PARALLEL_SIZE=2
TEMPERATURE=0.6

ALPHA=0.8
DELTA=0.8

# Create snapshot directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SNAPSHOT_DIR="/share/yangxizhong/workspace/step_level_deepconf/output/logs/scripts/baseline_LiveCodeBench-dpsk-distill-qwen3-8b_B64_snapshot_$TIMESTAMP"
mkdir -p "$SNAPSHOT_DIR"
# 1. Snapshot the Python script
TEMP_SCRIPT="$SNAPSHOT_DIR/selfstepconf_online_snapshot.py"
cp "$SCRIPT_PATH" "$TEMP_SCRIPT"
# 2. Snapshot the entire model directory
TEMP_MODEL_DIR="$SNAPSHOT_DIR/model"
cp -r "$MODEL_DIR" "$TEMP_MODEL_DIR"

DATASET_PATH1="/share/yangxizhong/data/LiveCodeBench_v5/lcbv5_test_questions.parquet"

LOG_DIR1="/share/yangxizhong/output/deepconf/baseline-dpsk/lcbv5-B64/LiveCodeBench-dpsk-distill-qwen3-8b"


LOG_FILE1="$LOG_DIR1/hmmt2025_LiveCodeBench-dpsk-distill-qwen3-8b_baseline_B64_T60.log"


mkdir -p "$(dirname "$LOG_FILE1")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=4,5 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR1 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE >> $LOG_FILE1 2>&1 &
PID1=$!
echo "LCB Process PID: $PID1"
wait $PID1
echo "LCB completed"


# ==================== 命令列表结束 ====================

echo "所有实验完成"