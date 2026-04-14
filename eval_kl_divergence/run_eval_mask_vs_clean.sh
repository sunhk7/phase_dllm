#!/bin/bash

CONFIGS=("16" "32" "64" "16,64")
PIDS=()

# 默认设置
MASK_RATIO="0.5"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mask_ratio) MASK_RATIO="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting Parallel Mask vs Clean Evaluation on Multiple GPUs..."

mkdir -p results/eval_mask_vs_clean

for i in "${!CONFIGS[@]}"; do
    WINDOW=${CONFIGS[$i]}
    
    # We module index by amount of gpus you have, e.g % 2 if you have GPUs 0, 1.
    # In run_parallel it iterates GPUs linearly by $i directly. Assuming user can scale.
    GPU_ID=$i

    TARGET_DIR="results/eval_mask_vs_clean/${MASK_RATIO}/${WINDOW}_${MASK_RATIO}"
    mkdir -p "$TARGET_DIR"

    echo "Launching Window=$WINDOW on GPU $GPU_ID in background..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_mask_vs_clean.py \
        --window "$WINDOW" \
        --mask_ratio $MASK_RATIO \
        --pos_id $GPU_ID \
        --num_samples 100 \
        > "${TARGET_DIR}/logs.txt" 2>&1 &

    PIDS+=($!)
done

echo "All GPUs are now computing independently! Output logs written to isolated subdirectories."

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "Multi-GPU Evaluation Completed!"
echo "Generating combined plots from the collected outputs..."

python eval_mask_vs_clean.py --plot_only --mask_ratio $MASK_RATIO

echo "Done! Check results/eval_mask_vs_clean/ for aggregated plots."
