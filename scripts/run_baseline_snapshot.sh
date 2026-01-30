#!/bin/bash

# Enhanced version with complete dependency snapshot
# This script creates snapshots of both the Python script and the model directory
# to ensure complete version consistency during execution

## CHECK:
# prompt
# datasets, script_path, model_dir, log_dir, beta, alpha, delta
# tensor_parallel_size, budget, temperature

# Configuration
DATASET_PATH="/share/yangxizhong/data/HMMT_2025/hmmt_25.jsonl"
# DATASET_PATH="/share/yangxizhong/data/GPQA-Diamond/test/gpqa_diamond.jsonl"
# DATASET_PATH="/share/yangxizhong/data/aime_2025/aime2025.jsonl"
# DATASET_PATH="/share/yangxizhong/data/aime_2024/data/train-00000-of-00001.parquet"
# DATASET_PATH="/share/yangxizhong/data/brumo_2025/data/train-00000-of-00001.parquet"

SCRIPT_PATH="/share/yangxizhong/workspace/step_level_deepconf/scripts/deepconf_dpsk_evaluate.py"
MODEL_DIR="/share/yangxizhong/workspace/step_level_deepconf/model"
LOG_DIR="/share/yangxizhong/workspace/step_level_deepconf/output/result/UpdateBaseline/hmmt2025-baseline-20251106/A100"


# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get the number of questions in the dataset
echo "Getting dataset length..."
DATASET_LENGTH=$(python3 -c "
import json
with open('$DATASET_PATH', 'r', encoding='utf-8') as file:
    data = [json.loads(line.strip()) for line in file]
print(len(data))
")

# echo "Getting dataset length..."
# DATASET_LENGTH=$(python3 -c "
# import pandas as pd
# df = pd.read_parquet('$DATASET_PATH')
# print(len(df))
# ")

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

# ===============================================================================
# EXECUTE WITH SNAPSHOT ENVIRONMENT
# ===============================================================================
echo "Starting execution with snapshot environment..." | tee -a "$MAIN_LOG"
# Loop through all questions using the snapshot
for qid in $(seq 0 $((DATASET_LENGTH - 1))); do
    echo "Processing question $qid/$((DATASET_LENGTH - 1)) at $(date)" | tee -a "$MAIN_LOG"

    # Create individual log file for this question
    QUESTION_LOG="$LOG_DIR/qid_${qid}_${TIMESTAMP}.log"

    # Run using the snapshot script (which now points to snapshot model directory)
    python3 "$TEMP_SCRIPT" \
        --qid "$qid" \
        --dataset "$DATASET_PATH" \
        --output_dir "$LOG_DIR" \
        --tensor_parallel_size 1 \
        --temperature 0.0 \
        2>&1 | tee "$QUESTION_LOG"

    # Check exit status
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Question $qid completed successfully" | tee -a "$MAIN_LOG"
    else
        echo "Question $qid failed with exit code $EXIT_CODE" | tee -a "$MAIN_LOG"
        # Optionally continue or break on error
        # break  # Uncomment to stop on first error
    fi

    echo "---" | tee -a "$MAIN_LOG"
done