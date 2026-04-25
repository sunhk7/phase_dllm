#!/bin/bash

# Configuration Dimensions: bash run.sh [max-new-tokens] [block-length] [w]
MAX_NEW_TOKENS=${1:-256}
BLOCK_LENGTH=${2:-32}
W_SIZE=${3:-8}
SHIFT_SIZE=$(($W_SIZE / 2))

OUT_DIR="results/${MAX_NEW_TOKENS}/${BLOCK_LENGTH}/${W_SIZE}"
mkdir -p "$OUT_DIR"

echo "Starting parallel experiments across 3 GPUs..."
echo "Output Directory: $OUT_DIR"
echo "Configuration: max_tokens=${MAX_NEW_TOKENS}, block_length=${BLOCK_LENGTH}, w=${W_SIZE}, shift_size=${SHIFT_SIZE}"
echo "------------------------------------------------------"

# Run baseline on GPU 4
echo "[GPU 4] Launching Baseline... (logs: $OUT_DIR/log_baseline.txt)"
CUDA_VISIBLE_DEVICES=4 python generate.py \
    --attention-mode baseline \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens $MAX_NEW_TOKENS \
    --block-length $BLOCK_LENGTH \
    --output-dir "$OUT_DIR" \
    --export-json > "$OUT_DIR/log_baseline.txt" 2>&1 &
P0=$!

# Run local_window on GPU 5
echo "[GPU 5] Launching Local Window... (logs: $OUT_DIR/log_local_window.txt)"
CUDA_VISIBLE_DEVICES=5 python generate.py \
    --attention-mode local_window \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens $MAX_NEW_TOKENS \
    --block-length $BLOCK_LENGTH \
    --output-dir "$OUT_DIR" \
    --export-json > "$OUT_DIR/log_local_window.txt" 2>&1 &
P1=$!

# Run swin_window on GPU 6 (Formerly swin_window_pad on GPU 7)
echo "[GPU 6] Launching Swin Window... (logs: $OUT_DIR/log_swin_window.txt)"
CUDA_VISIBLE_DEVICES=6 python generate.py \
    --attention-mode swin_window \
    --local-window-size $W_SIZE \
    --shift-size $SHIFT_SIZE \
    --benchmark-repeat 30 \
    --max-new-tokens $MAX_NEW_TOKENS \
    --block-length $BLOCK_LENGTH \
    --output-dir "$OUT_DIR" \
    --export-json > "$OUT_DIR/log_swin_window.txt" 2>&1 &
P2=$!

# Wait for all background processes to finish
wait $P0
wait $P1
wait $P2

echo "------------------------------------------------------"
echo "All 3 parallel inferences completed!"
echo "Generating visual comparison plots into $OUT_DIR..."
python plot_utils.py --w $W_SIZE --output-dir "$OUT_DIR"

echo "Check out your plots and logs at: $OUT_DIR/"
