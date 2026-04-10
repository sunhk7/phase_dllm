#!/bin/bash

# Configuration
# This will map these 4 specific window configurations to GPU 0, 1, 2, and 3
CONFIGS=("16" "32" "16,64")
PIDS=()

echo "🚀 Starting Parallel KL Divergence Evaluation on Multiple GPUs..."

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    # Assuming GPUs 0, 1, 2, 3 are available. Modification can be done here.
    GPU_ID=$i 
    
    echo "▶️ Launching Window=$CONFIG on GPU $GPU_ID in the background..."
    
    # Run independently and send to background
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_kl_divergence.py --window "$CONFIG" --pos_id $i &
    
    # Store process ID
    PIDS+=($!)
done

echo "⏳ All tasks launched. Waiting for them to complete (this may take a while)..."

# Wait for all background tasks to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "✅ All evaluation tasks have finished successfully!"

echo "📊 Generating combined plot from results..."
python eval_kl_divergence.py --plot_only

echo "🎉 Done! Check results/eval_kl_divergence/ for output."
