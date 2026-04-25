#!/bin/bash

# ============================================================
# Usage:  bash run.sh [max-new-tokens] [block-length] [w]
# 
# 5 experiments on 5 GPUs:
#   GPU 0: baseline
#   GPU 1: local_window
#   GPU 2: swin_window
#   GPU 3: swin_window + torch.compile
#   GPU 4: swin_triton  (fused Triton kernel)
# ============================================================

MAX_NEW_TOKENS=${1:-256}
BLOCK_LENGTH=${2:-256}
W_SIZE=${3:-32}
SHIFT_SIZE=$(($W_SIZE / 2))

OUT_DIR="results/${MAX_NEW_TOKENS}/${BLOCK_LENGTH}/${W_SIZE}"
mkdir -p "$OUT_DIR"

echo "Starting 5 parallel experiments..."
echo "Output Directory: $OUT_DIR"
echo "Config: max_tokens=${MAX_NEW_TOKENS}, block_length=${BLOCK_LENGTH}, w=${W_SIZE}, shift=${SHIFT_SIZE}"
echo "------------------------------------------------------"

CMD_COMMON="--local-window-size $W_SIZE --shift-size $SHIFT_SIZE --benchmark-repeat 30 --max-new-tokens $MAX_NEW_TOKENS --block-length $BLOCK_LENGTH --output-dir $OUT_DIR --export-json"

echo "[GPU 0] baseline"
CUDA_VISIBLE_DEVICES=0 python generate.py --attention-mode baseline $CMD_COMMON > "$OUT_DIR/log_baseline.txt" 2>&1 &
P0=$!

echo "[GPU 1] local_window"
CUDA_VISIBLE_DEVICES=1 python generate.py --attention-mode local_window $CMD_COMMON > "$OUT_DIR/log_local_window.txt" 2>&1 &
P1=$!

echo "[GPU 2] swin_window"
CUDA_VISIBLE_DEVICES=2 python generate.py --attention-mode swin_window $CMD_COMMON > "$OUT_DIR/log_swin_window.txt" 2>&1 &
P2=$!

echo "[GPU 3] swin_window + compile"
CUDA_VISIBLE_DEVICES=3 python generate.py --attention-mode swin_window --compile $CMD_COMMON > "$OUT_DIR/log_swin_window_compiled.txt" 2>&1 &
P3=$!

echo "[GPU 4] swin_triton"
CUDA_VISIBLE_DEVICES=4 python generate.py --attention-mode swin_triton $CMD_COMMON > "$OUT_DIR/log_swin_triton.txt" 2>&1 &
P4=$!

wait $P0 $P1 $P2 $P3 $P4

echo "------------------------------------------------------"
echo "All 5 experiments completed!"
echo "Generating plots..."
python plot_utils.py --w $W_SIZE --output-dir "$OUT_DIR"
echo "Done! Results at: $OUT_DIR/"
