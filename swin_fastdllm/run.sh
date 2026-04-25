#!/bin/bash

# Allows overriding window size via terminal parameter! e.g., `bash run.sh 16`
W_SIZE=${1:-8}
SHIFT_SIZE=$(($W_SIZE / 2))

mkdir -p results

echo "Starting parallel experiments across 3 GPUs..."
echo "Configuration: local-window-size = ${W_SIZE}, shift-size = ${SHIFT_SIZE}"
echo "------------------------------------------------------"

# Run baseline on GPU 0
echo "[GPU 0] Launching Baseline... (logs: results/log_baseline_w${W_SIZE}.txt)"
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --attention-mode baseline \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens 256 \
    --export-json > results/log_baseline_w${W_SIZE}.txt 2>&1 &
P0=$!

# Run local_window on GPU 1
echo "[GPU 1] Launching Local Window... (logs: results/log_local_window_w${W_SIZE}.txt)"
CUDA_VISIBLE_DEVICES=1 python generate.py \
    --attention-mode local_window \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens 256 \
    --export-json > results/log_local_window_w${W_SIZE}.txt 2>&1 &
P1=$!

# Run swin_window on GPU 2
echo "[GPU 2] Launching Swin Window... (logs: results/log_swin_window_w${W_SIZE}.txt)"
CUDA_VISIBLE_DEVICES=2 python generate.py \
    --attention-mode swin_window \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens 256 \
    --export-json > results/log_swin_window_w${W_SIZE}.txt 2>&1 &
P2=$!

# Run swin_window_pad on GPU 3
echo "[GPU 3] Launching Swin Pad... (logs: results/log_swin_window_pad_w${W_SIZE}.txt)"
CUDA_VISIBLE_DEVICES=3 python generate.py \
    --attention-mode swin_window_pad \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens 256 \
    --export-json > results/log_swin_window_pad_w${W_SIZE}.txt 2>&1 &
P3=$!

# Wait for all background processes to finish
wait $P0
wait $P1
wait $P2
wait $P3

echo "------------------------------------------------------"
echo "All 4 parallel inferences completed!"
echo "Generating visual comparison plots..."
python plot_utils.py --w $W_SIZE

echo "Check out images: results/throughput_w${W_SIZE}.png & results/memory_w${W_SIZE}.png"
