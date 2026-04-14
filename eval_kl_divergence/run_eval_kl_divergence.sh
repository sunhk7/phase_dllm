#!/bin/bash

# Configuration
# This will map these 4 specific window configurations to GPU 0, 1, 2, and 3
# (Fixed typo: replaced curly quotes on "64")
CONFIGS=("16" "32" "64" "16,64")
PIDS=()

echo "Starting Parallel KL Divergence Evaluation on Multiple GPUs..."

# Create log directory before bash attempts to redirect stdout
mkdir -p results/eval_kl_divergence

for i in "${!CONFIGS[@]}"; do
    WINDOW=${CONFIGS[$i]}
    GPU_ID=$i

    echo "Launching Window=$WINDOW on GPU $GPU_ID in background..."
    
    # We pass --pos_id for internal logic (even though stdout is caught, it prevents layout math errors)
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_kl_divergence.py \
        --window "$WINDOW" \
        --pos_id $GPU_ID \
        --num_samples 100 \
        > "results/eval_kl_divergence/logs_gpu_${GPU_ID}.txt" 2>&1 &

    # Store the process ID to wait for it later
    PIDS+=($!)
done

echo "All GPUs are now computing independently! Waiting for completion..."
echo "Note: Output logs strictly routed to results/eval_kl_divergence/logs_gpu_X.txt"

# Wait for all background tasks to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "Multi-GPU Evaluation Completed!"
echo "Generating combined plots from the collected outputs..."

python eval_kl_divergence.py --plot_only

echo "Done! Check results/eval_kl_divergence/ kl/ and jsd/ subfolders for arrays and root for plots."
