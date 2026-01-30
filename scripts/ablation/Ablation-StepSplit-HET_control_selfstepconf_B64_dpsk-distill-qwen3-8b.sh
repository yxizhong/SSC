#!/bin/bash

# experiment_queue.sh - 简洁的实验队列

echo "开始执行实验队列..."

# ==================== 实验命令列表 ====================
# 在这里添加你的bash命令，每行一个

BASH_PATH=/share/yangxizhong/workspace/step_level_deepconf/scripts/run_selfstepconf_snapshot.sh

SCRIPT_PATH="/share/yangxizhong/workspace/step_level_deepconf/output/logs/scripts/selfstepconf_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_B64_snapshot_20260102_191649/selfstepconf_online_snapshot.py"
MODEL_DIR="/share/yangxizhong/workspace/step_level_deepconf/output/logs/scripts/selfstepconf_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_B64_snapshot_20260102_191649/model"
MODEL_PATH="/share/yangxizhong/ckpt/DeepSeek-R1-0528-Qwen3-8B"
BUDGET=64
TENSOR_PARALLEL_SIZE=2
TEMPERATURE=0.6

# Create snapshot directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SNAPSHOT_DIR="/share/yangxizhong/workspace/step_level_deepconf/output/logs/scripts/selfstepconf_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_B64_snapshot_$TIMESTAMP"
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

LOG_DIR1="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/Ablation-StepSplit-HET-dpsk-distill-qwen3-8b"
LOG_DIR2="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2024-B64-95-80-80-20251203/Ablation-StepSplit-HET-dpsk-distill-qwen3-8b"
LOG_DIR3="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2025-B64-95-80-80-20251203/Ablation-StepSplit-HET-dpsk-distill-qwen3-8b"
LOG_DIR4="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/brumo2025-B64-95-80-80-20251204/Ablation-StepSplit-HET-dpsk-distill-qwen3-8b"
LOG_DIR5="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/gpqad-B64-95-80-80-20251202/Ablation-StepSplit-HET-dpsk-distill-qwen3-8b"

LOG_FILE1="$LOG_DIR1/hmmt2025_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE2="$LOG_DIR2/aime2024_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE3="$LOG_DIR3/aime2025_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE4="$LOG_DIR4/brumo2025_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"
LOG_FILE5="$LOG_DIR5/gpqad_Ablation-StepSplit-HET-dpsk-distill-qwen3-8b_selfstepconf_B64_T60.log"

mkdir -p "$(dirname "$LOG_FILE1")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=2,3 bash $BASH_PATH --dataset-path $DATASET_PATH1 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR1 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE >> $LOG_FILE1 2>&1 &
PID1=$!
echo "HMMT2025 Process PID: $PID1"
wait $PID1
echo "HMMT2025 completed"

mkdir -p "$(dirname "$LOG_FILE2")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=2,3 bash $BASH_PATH --dataset-path $DATASET_PATH2 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR2 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE >> $LOG_FILE2 2>&1 &
PID2=$!
echo "AIME2024 Process PID: $PID2"
wait $PID2
echo "AIME2024 completed"

mkdir -p "$(dirname "$LOG_FILE3")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=2,3 bash $BASH_PATH --dataset-path $DATASET_PATH3 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR3 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE >> $LOG_FILE3 2>&1 &
PID3=$!
echo "AIME2025 Process PID: $PID3"
wait $PID3
echo "AIME2025 completed"

mkdir -p "$(dirname "$LOG_FILE4")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=2,3 bash $BASH_PATH --dataset-path $DATASET_PATH4 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR4 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE >> $LOG_FILE4 2>&1 &
PID4=$!
echo "BRUMO2025 Process PID: $PID4"
wait $PID4
echo "BRUMO2025 completed"

mkdir -p "$(dirname "$LOG_FILE5")"
echo "Starting process at $(date), PID will be recorded..."
CUDA_VISIBLE_DEVICES=2,3 bash $BASH_PATH --dataset-path $DATASET_PATH5 --model-path $MODEL_PATH --script-path $TEMP_SCRIPT --model-dir $TEMP_MODEL_DIR --log-dir $LOG_DIR5 --budget $BUDGET --tensor-parallel-size $TENSOR_PARALLEL_SIZE --temperature $TEMPERATURE >> $LOG_FILE5 2>&1 &
PID5=$!
echo "GPQAD Process PID: $PID5"
wait $PID5
echo "GPQAD completed"
# ==================== 命令列表结束 ====================

echo "所有实验完成"