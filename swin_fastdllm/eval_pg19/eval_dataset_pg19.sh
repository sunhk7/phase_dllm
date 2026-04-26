#!/bin/bash

# ============================================================
# PG19 数据集评估：速度 + 质量
#
# Usage:  bash eval_pg19/eval_dataset_pg19.sh [seq_len] [block_length] [w] [num_samples]
#   (从项目根目录运行)
#
# 4 modes on 4 GPUs:
#   GPU 0: baseline
#   GPU 1: local_window
#   GPU 2: swin_window
#   GPU 3: swin_triton
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEQ_LEN=${1:-512}
BLOCK_LENGTH=${2:-256}
W_SIZE=${3:-32}
NUM_SAMPLES=${4:-10}

OUT_DIR="$SCRIPT_DIR/results/${SEQ_LEN}/${BLOCK_LENGTH}/${W_SIZE}"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " PG19 Evaluation"
echo " seq_len=$SEQ_LEN, block=$BLOCK_LENGTH, w=$W_SIZE"
echo " num_samples=$NUM_SAMPLES"
echo " output: $OUT_DIR"
echo "=========================================="

EVAL_SCRIPT="$SCRIPT_DIR/eval_dataset_pg19.py"
PLOT_SCRIPT="$SCRIPT_DIR/plot_eval_pg19.py"
CMD_COMMON="--seq-len $SEQ_LEN --block-length $BLOCK_LENGTH --w $W_SIZE --num-samples $NUM_SAMPLES --output-dir $OUT_DIR --warmup 2"

echo "[GPU 0] baseline"
CUDA_VISIBLE_DEVICES=0 python "$EVAL_SCRIPT" --attention-mode baseline $CMD_COMMON > "$OUT_DIR/log_baseline.txt" 2>&1 &
P0=$!

echo "[GPU 1] local_window"
CUDA_VISIBLE_DEVICES=1 python "$EVAL_SCRIPT" --attention-mode local_window $CMD_COMMON > "$OUT_DIR/log_local_window.txt" 2>&1 &
P1=$!

echo "[GPU 2] swin_window"
CUDA_VISIBLE_DEVICES=2 python "$EVAL_SCRIPT" --attention-mode swin_window $CMD_COMMON > "$OUT_DIR/log_swin_window.txt" 2>&1 &
P2=$!

echo "[GPU 3] swin_triton"
CUDA_VISIBLE_DEVICES=3 python "$EVAL_SCRIPT" --attention-mode swin_triton $CMD_COMMON > "$OUT_DIR/log_swin_triton.txt" 2>&1 &
P3=$!

wait $P0 $P1 $P2 $P3

echo ""
echo "All evaluations completed! Generating plots..."
python "$PLOT_SCRIPT" --output-dir "$OUT_DIR" --w $W_SIZE
echo "Done! Results at: $OUT_DIR/"
