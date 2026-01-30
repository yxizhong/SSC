#!/bin/bash

# experiment_queue.sh - 简洁的实验队列

echo "开始执行实验队列..."

# ==================== 实验命令列表 ====================
# 在这里添加你的bash命令，每行一个

BASH_PATH=/share/yangxizhong/workspace/step_level_deepconf/scripts/run_selfstepconf_snapshot.sh

SCRIPT_PATH="/share/yangxizhong/workspace/step_level_deepconf/scripts/selfstepconf_online.py"
MODEL_DIR="/share/yangxizhong/workspace/step_level_deepconf/model"
MODEL_PATH="/share/yangxizhong/ckpt/DeepSeek-R1-0528-Qwen3-8B"
BUDGET=64
TENSOR_PARALLEL_SIZE=8
TEMPERATURE=0.6
ALPHA=0.6

# Create snapshot directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SNAPSHOT_DIR="/share/yangxizhong/workspace/step_level_deepconf/output/logs/scripts/selfstepconf_Ablation-Alpha-60-dpsk-distill-qwen3-8b_B64_snapshot_$TIMESTAMP"
mkdir -p "$SNAPSHOT_DIR"
# 1. Snapshot the Python script
TEMP_SCRIPT="$SNAPSHOT_DIR/selfstepconf_online_snapshot.py"
cp "$SCRIPT_PATH" "$TEMP_SCRIPT"
# 2. Snapshot the entire model directory
TEMP_MODEL_DIR="$SNAPSHOT_DIR/model"
cp -r "$MODEL_DIR" "$TEMP_MODEL_DIR"

DATASET_PATH1="/share/yangxizhong/data/HMMT_2025/hmmt_25.jsonl"
DATASET_PATH2="/share/yangxizhong/data/aime_2024/data/train-00000-of-00001.parquet"
DATASET_PATH3="/share/yangxizhong/data/aime_2025/aime2025.jsonl"
DATASET_PATH4="/share/yangxizhong/data/brumo_2025/data/train-00000-of-00001.parquet"
DATASET_PATH5="/share/yangxizhong/data/GPQA-Diamond/test/gpqa_diamond.jsonl"

LOG_DIR1="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-10-dpsk-distill-qwen3-8b"
LOG_DIR2="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-20-dpsk-distill-qwen3-8b"
LOG_DIR3="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-30-dpsk-distill-qwen3-8b"
LOG_DIR4="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-40-dpsk-distill-qwen3-8b"
LOG_DIR5="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-50-dpsk-distill-qwen3-8b"
LOG_DIR6="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-60-dpsk-distill-qwen3-8b"
LOG_DIR7="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-70-dpsk-distill-qwen3-8b"
LOG_DIR8="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-Alpha-60-All-Delta-dpsk-distill-qwen3-8b/Ablation-Alpha-60-Delta-90-dpsk-distill-qwen3-8b"


LOG_FILE1="$LOG_DIR1/hmmt2025_Ablation-Alpha-60-Delta-10-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE2="$LOG_DIR2/hmmt2025_Ablation-Alpha-60-Delta-20-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE3="$LOG_DIR3/hmmt2025_Ablation-Alpha-60-Delta-30-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE4="$LOG_DIR4/hmmt2025_Ablation-Alpha-60-Delta-40-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE5="$LOG_DIR5/hmmt2025_Ablation-Alpha-60-Delta-50-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE6="$LOG_DIR6/hmmt2025_Ablation-Alpha-60-Delta-60-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE7="$LOG_DIR7/hmmt2025_Ablation-Alpha-60-Delta-70-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE8="$LOG_DIR8/hmmt2025_Ablation-Alpha-60-Delta-90-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"

# mkdir -p "$(dirname "$LOG_FILE1")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR1 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.1 >> $LOG_FILE1 2>&1 &
# PID1=$!
# echo "0.1x0.1 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"


# mkdir -p "$(dirname "$LOG_FILE2")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR2 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.2 >> $LOG_FILE2 2>&1 &
# PID1=$!
# echo "0.1x0.2 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"

# mkdir -p "$(dirname "$LOG_FILE3")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=4,5 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR3 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.3 >> $LOG_FILE3 2>&1 &
# PID1=$!
# echo "0.1x0.3 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"

# mkdir -p "$(dirname "$LOG_FILE4")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=4,5 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR4 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.4 >> $LOG_FILE4 2>&1 &
# PID1=$!
# echo "0.1x0.4 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"

# mkdir -p "$(dirname "$LOG_FILE5")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR5 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.5 >> $LOG_FILE5 2>&1 &
# PID1=$!
# echo "0.1x0.5 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"

# mkdir -p "$(dirname "$LOG_FILE6")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR6 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.6 >> $LOG_FILE6 2>&1 &
# PID1=$!
# echo "0.1x0.6 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"

mkdir -p "$(dirname "$LOG_FILE7")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR7 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.7 >> $LOG_FILE7 2>&1 &
PID1=$!
echo "0.1x0.7 HMMT2025 Process PID: $PID1"
wait $PID1
echo "HMMT2025 completed"

# mkdir -p "$(dirname "$LOG_FILE8")"
# echo "Starting process at $(date), PID will be recorded..."
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR8 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE --alpha $ALPHA --delta 0.9 >> $LOG_FILE8 2>&1 &
# PID1=$!
# echo "0.1x0.9 HMMT2025 Process PID: $PID1"
# wait $PID1
# echo "HMMT2025 completed"


# ==================== 命令列表结束 ====================

echo "所有实验完成"