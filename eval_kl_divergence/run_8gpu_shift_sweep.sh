#!/bin/bash
# 8-GPU SHiFT-dLLM Ablation Grid Search

echo "🚀 Starting 8-GPU SHiFT Simulation Grid Search..."

# Create log directory before bash attempts to redirect stdout
mkdir -p results/eval_shift_simulation

# 前 4 张卡跑严苛的 Gt (0.01)，后 4 张卡跑适中的 Gt (0.05)。各自涵盖 16->128 的窗口跨度测试！
CONFIGS=(
  "16:0.01"   # GPU 0: Window 16, Thresh 0.01
  "32:0.01"   # GPU 1: Window 32, Thresh 0.01
  "64:0.01"   # GPU 2: Window 64, Thresh 0.01
  "128:0.01"  # GPU 3: Window 128, Thresh 0.01
  "16:0.05"   # GPU 4: Window 16, Thresh 0.05
  "32:0.05"   # GPU 5: Window 32, Thresh 0.05
  "64:0.05"   # GPU 6: Window 64, Thresh 0.05
  "128:0.05"  # GPU 7: Window 128, Thresh 0.05
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
