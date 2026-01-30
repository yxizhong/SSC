#!/bin/bash

# Enhanced version with complete dependency snapshot
# This script creates snapshots of both the Python script and the model directory
# to ensure complete version consistency during execution

## CHECK:
# prompt
# datasets, script_path, model_dir, log_dir, beta, alpha, delta
# tensor_parallel_size, budget, temperature

# Configuration
# MODEL_PATH="/share/yangxizhong/ckpt/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_PATH="/share/yangxizhong/ckpt/DeepSeek-R1-Distill-Qwen-7B"
MODEL_PATH="/share/yangxizhong/ckpt/Qwen3-8B"

DATASET_PATH="/share/yangxizhong/data/GPQA-Diamond/test/gpqa_diamond.jsonl"
# DATASET_PATH="/share/yangxizhong/data/HMMT_2025/hmmt_25.jsonl"
# DATASET_PATH="/share/yangxizhong/data/brumo_2025/data/train-00000-of-00001.parquet"
# DATASET_PATH="/share/yangxizhong/data/aime_2024/data/train-00000-of-00001.parquet"
# DATASET_PATH="/share/yangxizhong/data/aime_2025/aime2025.jsonl"

SCRIPT_PATH="/share/yangxizhong/workspace/step_level_deepconf/scripts/selfstepconf_online.py"
MODEL_DIR="/share/yangxizhong/workspace/step_level_deepconf/model"
LOG_DIR="/share/yangxizhong/output/deepconf/selfstepconf_deepthink/gpqad-B64-95-80-80-20251202/Qwen3-8B-NonThinking"

BETA=0.95
ALPHA=0.8
DELTA=0.8

# WARMUP_TRACES_DIR="/share/yangxizhong/output/deepconf/baseline-dpsk/pass512-20251106"
WARMUP_TRACES_DIR=" "

PARALLEL=0
BUDGET=64
TEMPERATURE=0.7
TENSOR_PARALLEL_SIZE=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --script-path)
            SCRIPT_PATH="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --delta)
            DELTA="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get the number of questions in the dataset
echo "Getting dataset length..."
if [[ "$DATASET_PATH" == *.jsonl ]]; then
    echo "Detected JSONL format, using JSON processing..."
    DATASET_LENGTH=$(python3 -c "
import json
with open('$DATASET_PATH', 'r', encoding='utf-8') as file:
    data = [json.loads(line.strip()) for line in file]
print(len(data))
")
elif [[ "$DATASET_PATH" == *.parquet ]]; then
    echo "Detected Parquet format, using pandas processing..."
    DATASET_LENGTH=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$DATASET_PATH')
print(len(df))
")
else
    echo "Error: Unsupported file format. Only .jsonl and .parquet are supported."
    exit 1
fi
echo "Dataset contains $DATASET_LENGTH questions"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_all_baseline_$TIMESTAMP.log"

echo "Starting batch processing at $(date)" | tee "$MAIN_LOG"
echo "Processing $DATASET_LENGTH questions" | tee -a "$MAIN_LOG"

# ===============================================================================
# CREATE COMPLETE SNAPSHOT ENVIRONMENT
# ===============================================================================

echo "Creating complete snapshot environment..." | tee -a "$MAIN_LOG"

# Create snapshot directory
SNAPSHOT_DIR="$LOG_DIR/snapshot_$TIMESTAMP"
mkdir -p "$SNAPSHOT_DIR"

# 1. Snapshot the Python script
TEMP_SCRIPT="$SNAPSHOT_DIR/selfstepconf_online_snapshot.py"
cp "$SCRIPT_PATH" "$TEMP_SCRIPT"

# 2. Snapshot the entire model directory
TEMP_MODEL_DIR="$SNAPSHOT_DIR/model"
cp -r "$MODEL_DIR" "$TEMP_MODEL_DIR"

echo "Created script snapshot: $TEMP_SCRIPT" | tee -a "$MAIN_LOG"
echo "Created model directory snapshot: $TEMP_MODEL_DIR" | tee -a "$MAIN_LOG"

# 3. Modify the snapshot script to use the snapshot model directory
# Replace the sys.path.insert line to point to snapshot directory
sed -i "s|sys.path.insert(0, \"/share/yangxizhong/workspace/step_level_deepconf\")|sys.path.insert(0, \"$SNAPSHOT_DIR\")|g" "$TEMP_SCRIPT"

# Also handle any other hardcoded paths in the script if they exist
# This ensures the script uses the snapshot model directory
echo "Modified script to use snapshot model directory" | tee -a "$MAIN_LOG"

# 4. Verify the snapshot was created correctly
if [ ! -f "$TEMP_SCRIPT" ]; then
    echo "ERROR: Failed to create script snapshot" | tee -a "$MAIN_LOG"
    exit 1
fi

if [ ! -d "$TEMP_MODEL_DIR" ]; then
    echo "ERROR: Failed to create model directory snapshot" | tee -a "$MAIN_LOG"
    exit 1
fi

# 5. Calculate checksums for verification
SCRIPT_CHECKSUM=$(md5sum "$TEMP_SCRIPT" | cut -d' ' -f1)
MODEL_CHECKSUM=$(find "$TEMP_MODEL_DIR" -type f -exec md5sum {} \; | md5sum | cut -d' ' -f1)

echo "Script snapshot checksum: $SCRIPT_CHECKSUM" | tee -a "$MAIN_LOG"
echo "Model directory checksum: $MODEL_CHECKSUM" | tee -a "$MAIN_LOG"

# ===============================================================================
# EXECUTE WITH SNAPSHOT ENVIRONMENT
# ===============================================================================

echo "Starting execution with snapshot environment..." | tee -a "$MAIN_LOG"

# Loop through all questions using the snapshot
if [[ "$PARALLEL" == "1" ]]; then
    echo "Parallel mode enabled" | tee -a "$MAIN_LOG"
    
    # 从CUDA_VISIBLE_DEVICES获取可用GPU列表
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        # 解析CUDA_VISIBLE_DEVICES
        IFS=',' read -ra AVAILABLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
        GPU_COUNT=${#AVAILABLE_GPUS[@]}
        echo "Using GPUs specified by CUDA_VISIBLE_DEVICES: ${AVAILABLE_GPUS[*]}" | tee -a "$MAIN_LOG"
    else
        # 如果没有设置CUDA_VISIBLE_DEVICES，获取所有可用GPU
        TOTAL_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        AVAILABLE_GPUS=($(seq 0 $((TOTAL_GPU_COUNT - 1))))
        GPU_COUNT=$TOTAL_GPU_COUNT
        echo "CUDA_VISIBLE_DEVICES not set, using all $GPU_COUNT GPUs: ${AVAILABLE_GPUS[*]}" | tee -a "$MAIN_LOG"
    fi
    
    if [[ $GPU_COUNT -eq 0 ]]; then
        echo "No GPUs available, falling back to sequential processing" | tee -a "$MAIN_LOG"
        PARALLEL=0
    else
        echo "Will use $GPU_COUNT GPUs for parallel processing" | tee -a "$MAIN_LOG"
        
        # 并行处理函数
        process_gpu_batch() {
            local gpu_index=$1  # 这是在AVAILABLE_GPUS数组中的索引
            local actual_gpu_id=${AVAILABLE_GPUS[$gpu_index]}  # 实际的GPU ID
            shift
            local questions=("$@")
            
            # 设置这个进程只能看到一个GPU，并且是相对索引0
            export CUDA_VISIBLE_DEVICES=$actual_gpu_id
            
            for qid in "${questions[@]}"; do
                echo "GPU $actual_gpu_id (index $gpu_index): Processing question $qid at $(date)" | tee -a "$MAIN_LOG"
                
                QUESTION_LOG="$LOG_DIR/qid_${qid}_gpu${actual_gpu_id}_${TIMESTAMP}.log"
                
                python3 "$TEMP_SCRIPT" \
                    --model "$MODEL_PATH" \
                    --qid "$qid" \
                    --rid "$qid" \
                    --dataset "$DATASET_PATH" \
                    --output_dir "$LOG_DIR" \
                    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
                    --beta $BETA \
                    --alpha $ALPHA \
                    --delta $DELTA \
                    --budget $BUDGET \
                    --temperature $TEMPERATURE \
                    --warmup_traces_dir "$WARMUP_TRACES_DIR" \
                    2>&1 | tee "$QUESTION_LOG"
                
                EXIT_CODE=${PIPESTATUS[0]}
                if [ $EXIT_CODE -eq 0 ]; then
                    echo "GPU $actual_gpu_id: Question $qid completed successfully" | tee -a "$MAIN_LOG"
                else
                    echo "GPU $actual_gpu_id: Question $qid failed with exit code $EXIT_CODE" | tee -a "$MAIN_LOG"
                fi
            done
        }
        
        # 分配任务到各个GPU
        PIDS=()
        for gpu_index in $(seq 0 $((GPU_COUNT - 1))); do
            # 计算这个GPU应该处理的问题范围
            questions_per_gpu=$((DATASET_LENGTH / GPU_COUNT))
            start_qid=$((gpu_index * questions_per_gpu))
            
            if [[ $gpu_index -eq $((GPU_COUNT - 1)) ]]; then
                # 最后一个GPU处理剩余的所有问题
                end_qid=$((DATASET_LENGTH - 1))
            else
                end_qid=$((start_qid + questions_per_gpu - 1))
            fi
            
            if [[ $start_qid -le $end_qid ]]; then
                gpu_questions=($(seq $start_qid $end_qid))
                echo "Assigning questions $start_qid-$end_qid to GPU ${AVAILABLE_GPUS[$gpu_index]}" | tee -a "$MAIN_LOG"
                process_gpu_batch $gpu_index "${gpu_questions[@]}" &
                PIDS+=($!)
            fi
        done
        
        # 等待所有GPU完成
        echo "Waiting for all GPU processes to complete..." | tee -a "$MAIN_LOG"
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        
        echo "Parallel processing completed" | tee -a "$MAIN_LOG"
    fi
fi

# 如果不是并行模式或者GPU检测失败，使用原始的顺序处理
if [[ "$PARALLEL" != "1" ]]; then
    echo "Sequential processing mode" | tee -a "$MAIN_LOG"
    
    for qid in $(seq 0 $((DATASET_LENGTH - 1))); do
        echo "Processing question $qid/$((DATASET_LENGTH - 1)) at $(date)" | tee -a "$MAIN_LOG"

        QUESTION_LOG="$LOG_DIR/qid_${qid}_${TIMESTAMP}.log"

        python3 "$TEMP_SCRIPT" \
            --model "$MODEL_PATH" \
            --qid "$qid" \
            --rid "$qid" \
            --dataset "$DATASET_PATH" \
            --output_dir "$LOG_DIR" \
            --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
            --beta $BETA \
            --alpha $ALPHA \
            --delta $DELTA \
            --budget $BUDGET \
            --temperature $TEMPERATURE \
            --warmup_traces_dir "$WARMUP_TRACES_DIR" \
            2>&1 | tee "$QUESTION_LOG"

        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Question $qid completed successfully" | tee -a "$MAIN_LOG"
        else
            echo "Question $qid failed with exit code $EXIT_CODE" | tee -a "$MAIN_LOG"
        fi

        echo "---" | tee -a "$MAIN_LOG"
    done
fi

# ===============================================================================
# COMPLETION AND CLEANUP
# ===============================================================================

echo "Batch processing completed at $(date)" | tee -a "$MAIN_LOG"
echo "Main log: $MAIN_LOG"
echo "Individual logs: $LOG_DIR/qid_*_${TIMESTAMP}.log"
echo "Snapshot directory: $SNAPSHOT_DIR"
echo "Script snapshot: $TEMP_SCRIPT (checksum: $SCRIPT_CHECKSUM)"
echo "Model snapshot: $TEMP_MODEL_DIR (checksum: $MODEL_CHECKSUM)"

# Create snapshot info file for future reference
SNAPSHOT_INFO="$SNAPSHOT_DIR/snapshot_info.json"
cat > "$SNAPSHOT_INFO" << EOF
{
    "timestamp": "$TIMESTAMP",
    "original_script": "$SCRIPT_PATH",
    "original_model_dir": "$MODEL_DIR",
    "snapshot_script": "$TEMP_SCRIPT",
    "snapshot_model_dir": "$TEMP_MODEL_DIR",
    "script_checksum": "$SCRIPT_CHECKSUM",
    "model_checksum": "$MODEL_CHECKSUM",
    "execution_log": "$MAIN_LOG",
    "parameters": {
        "beta": $BETA,
        "alpha": $ALPHA,
        "delta": $DELTA,
        "dataset": "$DATASET_PATH"
    }
}
EOF

echo "Snapshot information saved to: $SNAPSHOT_INFO"

# Optional: Archive the snapshot for long-term storage
# Uncomment the following lines if you want to create a compressed archive
# echo "Creating snapshot archive..."
# tar -czf "$LOG_DIR/snapshot_archive_$TIMESTAMP.tar.gz" -C "$LOG_DIR" "snapshot_$TIMESTAMP"
# echo "Snapshot archive created: $LOG_DIR/snapshot_archive_$TIMESTAMP.tar.gz"

echo "Execution completed with complete dependency isolation!"