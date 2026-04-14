#!/bin/bash
# 8-GPU SHiFT-dLLM Ablation Grid Search

echo "🚀 Starting 8-GPU SHiFT Simulation Grid Search..."

# Create log directory before bash attempts to redirect stdout
mkdir -p results/eval_shift_simulation


CONFIGS=(
  "16:0.01"   # GPU 0: Window 16, Thresh 0.01
  "32:0.01"   # GPU 1: Window 32, Thresh 0.01
)

PIDS=()

for i in "${!CONFIGS[@]}"; do
    IFS=":" read -r win thresh <<< "${CONFIGS[$i]}"
    GPU_ID=$i 
    
    echo "▶️ Launching Window=$win, Thresh=$thresh on GPU $GPU_ID in background..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_shift_gt_simulation.py \
        --window $win --threshold $thresh \
        > "results/eval_shift_simulation/logs_gpu_${GPU_ID}.txt" 2>&1 &
    
    PIDS+=($!)
done

echo "⏳ All 8 GPUs are now fully loaded and computing independently! Waiting for completion..."

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "✅ 8-GPU Ablation Grid Search Completed! All artifacts and plots are saved individually per condition."

echo "📊 Rendering 2x1 Top-1 Accuracy Subplots..."
python eval_shift_gt_simulation.py --plot_only
echo "🎉 Comprehensive Analysis Complete!"
