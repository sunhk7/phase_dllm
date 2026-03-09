#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"          # gsm8k | prompt | analysis | all
shift || true

CONFIG_PATH="config.yaml"
if [ "$#" -gt 0 ] && [[ "${1:-}" != -* ]]; then
  CONFIG_PATH="$1"
  shift
fi

EXTRA_ARGS=("$@")

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] config not found: $CONFIG_PATH"
  exit 1
fi

# ── 从 config.yaml 读取参数 ──
eval "$(python3 - "$CONFIG_PATH" <<'PY'
import shlex
import sys

try:
    import yaml
except Exception:
    raise SystemExit("[ERROR] PyYAML is required: pip install pyyaml")

cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))

def get(path, default=None):
    cur = cfg
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")

# Model
emit('MODEL_ID', get('model.model_id', 'GSAI-ML/LLaDA-8B-Instruct'))

# Paths
emit('RESULTS_DIR', get('paths.results_dir', 'results'))
emit('GSM8K_SUBDIR', get('paths.gsm8k_subdir', 'gsm8k'))
emit('PROMPT_SUBDIR', get('paths.prompt_subdir', 'prompt'))

# L-Shape Mask
emit('L_SHAPE_ENABLED', get('l_shape_mask.enabled', False))
emit('DYNAMIC_WINDOW_SIZE', get('l_shape_mask.dynamic_window_size', 64))

# Attention Recording
emit('RECORD_ATTENTION', get('attention_recording.record_attention', False))

# GSM8K
emit('GSM8K_ENABLED', get('gsm8k.enabled', False))
emit('GSM8K_SPLIT', get('gsm8k.split', 'test'))
emit('GSM8K_MAX_SAMPLES', get('gsm8k.max_samples', 10))
emit('GSM8K_BATCH_SIZE', get('gsm8k.batch_size', 1))
emit('GSM8K_STEPS', get('gsm8k.steps', 128))
emit('GSM8K_GEN_LENGTH', get('gsm8k.gen_length', 128))
emit('GSM8K_BLOCK_LENGTH', get('gsm8k.block_length', 128))
emit('GSM8K_TEMPERATURE', get('gsm8k.temperature', 0.0))
emit('GSM8K_CFG_SCALE', get('gsm8k.cfg_scale', 0.0))
emit('GSM8K_REMASKING', get('gsm8k.remasking', 'low_confidence'))
emit('GSM8K_LOCAL_HALF_WINDOW', get('gsm8k.local_half_window', 32))

# Prompt
emit('PROMPT_ENABLED', get('prompt.enabled', False))
emit('PROMPTS_FILE', get('prompt.prompts_file', 'prompts/prompts.txt'))
emit('PROMPT_KEY', get('prompt.prompt_key', 'prompt'))
emit('PROMPT_BATCH_SIZE', get('prompt.batch_size', 1))
emit('PROMPT_STEPS', get('prompt.steps', 128))
emit('PROMPT_GEN_LENGTH', get('prompt.gen_length', 128))
emit('PROMPT_BLOCK_LENGTH', get('prompt.block_length', 128))
emit('PROMPT_TEMPERATURE', get('prompt.temperature', 0.0))
emit('PROMPT_CFG_SCALE', get('prompt.cfg_scale', 0.0))
emit('PROMPT_REMASKING', get('prompt.remasking', 'low_confidence'))
emit('PROMPT_LOCAL_HALF_WINDOW', get('prompt.local_half_window', 32))

# Analysis
emit('ANALYSIS_WINDOW_SIZE', get('analysis.window_size', 64))
emit('ANALYSIS_OUTPUT', get('analysis.output_filename', 'entropy_vs_locality.png'))

# Runtime
emit('DEVICE', get('runtime.device', 'auto'))

# Run switches
emit('RUN_GSM8K', get('run.gsm8k', False))
emit('RUN_PROMPT', get('run.prompt', False))
emit('RUN_ANALYSIS', get('run.analysis', False))
PY
)"

GSM8K_RESULTS_DIR="${RESULTS_DIR}/${GSM8K_SUBDIR}"
PROMPT_RESULTS_DIR="${RESULTS_DIR}/${PROMPT_SUBDIR}"

mkdir -p "$GSM8K_RESULTS_DIR"
mkdir -p "$PROMPT_RESULTS_DIR"

# ────────────────────────────────────────────
#  GSM8K Generation (with optional L-Shape mask + attention recording)
# ────────────────────────────────────────────
run_gsm8k() {
  if [ "$RUN_GSM8K" != "True" ] && [ "$RUN_GSM8K" != "true" ]; then
    echo "[SKIP] run.gsm8k=false"
    return
  fi
  if [ "$GSM8K_ENABLED" != "True" ] && [ "$GSM8K_ENABLED" != "true" ]; then
    echo "[SKIP] gsm8k.enabled=false"
    return
  fi

  echo "════════════════════════════════════════════"
  echo "[RUN] GSM8K generation"
  echo "  L-Shape mask enabled: $L_SHAPE_ENABLED (W=$DYNAMIC_WINDOW_SIZE)"
  echo "  Record attention:     $RECORD_ATTENTION"
  echo "════════════════════════════════════════════"

  # Build optional L-Shape + attention args
  L_SHAPE_ARGS=()
  if [ "$L_SHAPE_ENABLED" = "True" ] || [ "$L_SHAPE_ENABLED" = "true" ]; then
    L_SHAPE_ARGS+=(--dynamic-window-size "$DYNAMIC_WINDOW_SIZE")
  fi
  if [ "$RECORD_ATTENTION" = "True" ] || [ "$RECORD_ATTENTION" = "true" ]; then
    L_SHAPE_ARGS+=(--record-attention)
  fi

  python3 generate.py \
    --model-id "$MODEL_ID" \
    --dataset gsm8k \
    --split "$GSM8K_SPLIT" \
    --max-samples "$GSM8K_MAX_SAMPLES" \
    --batch-size "$GSM8K_BATCH_SIZE" \
    --steps "$GSM8K_STEPS" \
    --gen-length "$GSM8K_GEN_LENGTH" \
    --block-length "$GSM8K_BLOCK_LENGTH" \
    --temperature "$GSM8K_TEMPERATURE" \
    --cfg-scale "$GSM8K_CFG_SCALE" \
    --remasking "$GSM8K_REMASKING" \
    --local-half-window "$GSM8K_LOCAL_HALF_WINDOW" \
    --results-dir "$RESULTS_DIR" \
    --device "$DEVICE" \
    "${L_SHAPE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  echo "[DONE] GSM8K outputs saved to $GSM8K_RESULTS_DIR"
}

# ────────────────────────────────────────────
#  Prompt Generation (with optional L-Shape mask + attention recording)
# ────────────────────────────────────────────
run_prompt() {
  if [ "$RUN_PROMPT" != "True" ] && [ "$RUN_PROMPT" != "true" ]; then
    echo "[SKIP] run.prompt=false"
    return
  fi
  if [ "$PROMPT_ENABLED" != "True" ] && [ "$PROMPT_ENABLED" != "true" ]; then
    echo "[SKIP] prompt.enabled=false"
    return
  fi

  echo "════════════════════════════════════════════"
  echo "[RUN] Prompt generation"
  echo "  L-Shape mask enabled: $L_SHAPE_ENABLED (W=$DYNAMIC_WINDOW_SIZE)"
  echo "  Record attention:     $RECORD_ATTENTION"
  echo "════════════════════════════════════════════"

  PROMPT_CMD_ARGS=(
    --model-id "$MODEL_ID"
    --batch-size "$PROMPT_BATCH_SIZE"
    --steps "$PROMPT_STEPS"
    --gen-length "$PROMPT_GEN_LENGTH"
    --block-length "$PROMPT_BLOCK_LENGTH"
    --temperature "$PROMPT_TEMPERATURE"
    --cfg-scale "$PROMPT_CFG_SCALE"
    --remasking "$PROMPT_REMASKING"
    --local-half-window "$PROMPT_LOCAL_HALF_WINDOW"
    --results-dir "$RESULTS_DIR"
    --device "$DEVICE"
  )

  if [ -f "$PROMPTS_FILE" ]; then
    PROMPT_CMD_ARGS+=(--prompts-file "$PROMPTS_FILE" --prompt-key "$PROMPT_KEY")
  else
    echo "[WARN] prompts file not found: $PROMPTS_FILE, using DEFAULT_PROMPTS"
  fi

  # Build optional L-Shape + attention args
  L_SHAPE_ARGS=()
  if [ "$L_SHAPE_ENABLED" = "True" ] || [ "$L_SHAPE_ENABLED" = "true" ]; then
    L_SHAPE_ARGS+=(--dynamic-window-size "$DYNAMIC_WINDOW_SIZE")
  fi
  if [ "$RECORD_ATTENTION" = "True" ] || [ "$RECORD_ATTENTION" = "true" ]; then
    L_SHAPE_ARGS+=(--record-attention)
  fi

  python3 generate_prompt.py "${PROMPT_CMD_ARGS[@]}" "${L_SHAPE_ARGS[@]}" "${EXTRA_ARGS[@]}"

  echo "[DONE] Prompt outputs saved to $PROMPT_RESULTS_DIR"
}

# ────────────────────────────────────────────
#  Offline Analysis: Entropy vs. Locality
# ────────────────────────────────────────────
run_analysis() {
  if [ "$RUN_ANALYSIS" != "True" ] && [ "$RUN_ANALYSIS" != "true" ]; then
    echo "[SKIP] run.analysis=false"
    return
  fi

  echo "════════════════════════════════════════════"
  echo "[RUN] Offline analysis: Entropy vs. Locality"
  echo "  Window size: $ANALYSIS_WINDOW_SIZE"
  echo "════════════════════════════════════════════"

  count=0
  for attn_pt in "$GSM8K_RESULTS_DIR"/attn_weights_*.pt "$PROMPT_RESULTS_DIR"/attn_weights_*.pt; do
    [ -f "$attn_pt" ] || continue

    # Extract prompt_length from companion meta file if available
    meta="${attn_pt%.pt}_meta.json"
    if [ -f "$meta" ]; then
      PROMPT_LEN="$(python3 -c "import json; print(json.load(open('$meta'))['prompt_length'])")"
    else
      echo "[WARN] No meta file for $attn_pt, using default prompt_length=128"
      PROMPT_LEN=128
    fi

    png="${attn_pt%.pt}_${ANALYSIS_OUTPUT}"

    python3 analyze_attention.py \
      --attn-path "$attn_pt" \
      --prompt-length "$PROMPT_LEN" \
      --window-size "$ANALYSIS_WINDOW_SIZE" \
      --output-path "$png"

    count=$((count + 1))
  done

  if [ "$count" -eq 0 ]; then
    echo "[WARN] No attn_weights_*.pt files found. Run generation with record_attention=true first."
    echo "[INFO] You can also run analysis manually:"
    echo "  python3 analyze_attention.py --attn-path <path.pt> --prompt-length <N> --window-size $ANALYSIS_WINDOW_SIZE"
  else
    echo "[DONE] Generated $count analysis plots"
  fi
}

# ── Main dispatch ──
case "$MODE" in
  gsm8k)
    run_gsm8k
    run_analysis
    ;;
  prompt)
    run_prompt
    run_analysis
    ;;
  analysis)
    run_analysis
    ;;
  all)
    run_gsm8k
    run_prompt
    run_analysis
    ;;
  *)
    echo "Usage: bash commands.sh [gsm8k|prompt|analysis|all] [config.yaml] [extra args...]"
    exit 1
    ;;
esac
