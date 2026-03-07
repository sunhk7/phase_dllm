#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"        # dummy | real | prompt | all
CONFIG_PATH="${2:-config.yaml}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] config not found: $CONFIG_PATH"
  exit 1
fi

# 从 config.yaml 读取参数（依赖 PyYAML）
eval "$(python3 - "$CONFIG_PATH" <<'PY'
import shlex
import sys

try:
    import yaml
except Exception as e:
    raise SystemExit("[ERROR] PyYAML is required: pip install pyyaml")

cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))

def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")

emit('RESULTS_DIR', cfg['paths']['results_dir'])
emit('MODEL_ID', cfg['model']['model_id'])

# dummy
emit('DUMMY_ENABLED', cfg['dummy']['enabled'])
emit('DUMMY_TOKENIZER', cfg['dummy']['tokenizer'])
emit('DUMMY_STEPS', cfg['dummy']['steps'])
emit('DUMMY_GEN_LENGTH', cfg['dummy']['gen_length'])
emit('DUMMY_BLOCK_LENGTH', cfg['dummy']['block_length'])
emit('DUMMY_LOCAL_HALF_WINDOW', cfg.get('dummy', {}).get('local_half_window', 32))
emit('DUMMY_DEVICE', cfg['dummy']['device'])
emit('DUMMY_NPY', cfg['dummy']['output_npy'])
emit('DUMMY_PNG', cfg['dummy']['output_png'])

# real
emit('REAL_ENABLED', cfg['real']['enabled'])
emit('REAL_DATASET', cfg['real']['dataset'])
emit('REAL_SPLIT', cfg['real']['split'])
emit('REAL_MAX_SAMPLES', cfg['real']['max_samples'])
emit('REAL_BATCH_SIZE', cfg['real']['batch_size'])
emit('REAL_STEPS', cfg['real']['steps'])
emit('REAL_GEN_LENGTH', cfg['real']['gen_length'])
emit('REAL_BLOCK_LENGTH', cfg['real']['block_length'])
emit('REAL_LOCAL_HALF_WINDOW', cfg.get('real', {}).get('local_half_window', 32))
emit('REAL_TEMPERATURE', cfg['real']['temperature'])
emit('REAL_CFG_SCALE', cfg['real']['cfg_scale'])
emit('REAL_REMASKING', cfg['real']['remasking'])
emit('REAL_DEVICE', cfg['real']['device'])
emit('REAL_TITLE_PREFIX', cfg['real']['heatmap_title_prefix'])

# prompt
emit('PROMPT_ENABLED', cfg['prompt']['enabled'])
emit('PROMPT_BATCH_SIZE', cfg['prompt']['batch_size'])
emit('PROMPT_STEPS', cfg['prompt']['steps'])
emit('PROMPT_GEN_LENGTH', cfg['prompt']['gen_length'])
emit('PROMPT_BLOCK_LENGTH', cfg['prompt']['block_length'])
emit('PROMPT_LOCAL_HALF_WINDOW', cfg.get('prompt', {}).get('local_half_window', 32))
emit('PROMPT_TEMPERATURE', cfg['prompt']['temperature'])
emit('PROMPT_CFG_SCALE', cfg['prompt']['cfg_scale'])
emit('PROMPT_REMASKING', cfg['prompt']['remasking'])
emit('PROMPT_DEVICE', cfg['prompt']['device'])
emit('PROMPT_TITLE_PREFIX', cfg['prompt']['heatmap_title_prefix'])

# longbench
emit('LONGBENCH_ENABLED', cfg.get('longbench', {}).get('enabled', False))
if 'longbench' in cfg:
    emit('LONGBENCH_DATASETS', ','.join(cfg['longbench']['datasets']))
    emit('LONGBENCH_SAMPLES_PER_DATASET', cfg['longbench']['samples_per_dataset'])
    emit('LONGBENCH_BATCH_SIZE', cfg['longbench']['batch_size'])
    emit('LONGBENCH_STEPS', cfg['longbench']['steps'])
    emit('LONGBENCH_GEN_LENGTH', cfg['longbench']['gen_length'])
    emit('LONGBENCH_BLOCK_LENGTH', cfg['longbench']['block_length'])
    emit('LONGBENCH_LOCAL_HALF_WINDOW', cfg.get('longbench', {}).get('local_half_window', 64))
    emit('LONGBENCH_TEMPERATURE', cfg['longbench']['temperature'])
    emit('LONGBENCH_CFG_SCALE', cfg['longbench']['cfg_scale'])
    emit('LONGBENCH_REMASKING', cfg['longbench']['remasking'])
    emit('LONGBENCH_DEVICE', cfg['longbench']['device'])
    emit('LONGBENCH_TITLE_PREFIX', cfg['longbench']['heatmap_title_prefix'])
PY
)"

mkdir -p "$RESULTS_DIR"
DATASET_RESULTS_DIR="$RESULTS_DIR/$REAL_DATASET"

run_dummy() {
  if [ "$DUMMY_ENABLED" != "True" ] && [ "$DUMMY_ENABLED" != "true" ]; then
    echo "[SKIP] dummy.enabled=false"
    return
  fi

  echo "[RUN] Dummy chain"
  python3 test_dummy_model.py \
    --tokenizer "$DUMMY_TOKENIZER" \
    --steps "$DUMMY_STEPS" \
    --gen-length "$DUMMY_GEN_LENGTH" \
    --block-length "$DUMMY_BLOCK_LENGTH" \
    --local-half-window "$DUMMY_LOCAL_HALF_WINDOW" \
    --device "$DUMMY_DEVICE" \
    --output "$DUMMY_NPY" \
    --plot "$DUMMY_PNG"
}

run_real() {
  if [ "$REAL_ENABLED" != "True" ] && [ "$REAL_ENABLED" != "true" ]; then
    echo "[SKIP] real.enabled=false"
    return
  fi

  echo "[RUN] Real generation chain"
  python3 generate.py \
    --model-id "$MODEL_ID" \
    --dataset "$REAL_DATASET" \
    --split "$REAL_SPLIT" \
    --max-samples "$REAL_MAX_SAMPLES" \
    --batch-size "$REAL_BATCH_SIZE" \
    --steps "$REAL_STEPS" \
    --gen-length "$REAL_GEN_LENGTH" \
    --block-length "$REAL_BLOCK_LENGTH" \
    --local-half-window "$REAL_LOCAL_HALF_WINDOW" \
    --temperature "$REAL_TEMPERATURE" \
    --cfg-scale "$REAL_CFG_SCALE" \
    --remasking "$REAL_REMASKING" \
    --results-dir "$RESULTS_DIR" \
    --device "$REAL_DEVICE"

  # Legacy single-file flow kept for reference (may be removed later).
  # python3 generate.py
  # if [ ! -f "$REAL_SOURCE_NPY" ]; then
  #   echo "[ERROR] Expected output not found: $REAL_SOURCE_NPY"
  #   exit 1
  # fi
  # mv -f "$REAL_SOURCE_NPY" "$REAL_NPY"
  # python3 plot_dynamics.py "$REAL_NPY" --output "$REAL_PNG" --title "$REAL_TITLE"

  mkdir -p "$DATASET_RESULTS_DIR"
  count=0
  for npy in "$DATASET_RESULTS_DIR"/gsm8k_dynamics_*.npy; do
    if [ ! -f "$npy" ]; then
      continue
    fi
    png="${npy%.npy}.png"
    title="$REAL_TITLE_PREFIX $(basename "$npy" .npy)"
    python3 plot_dynamics.py "$npy" --output "$png" --title "$title"
    count=$((count + 1))
  done

  if [ "$count" -eq 0 ]; then
    echo "[WARN] No GSM8K dynamics npy files found in $DATASET_RESULTS_DIR"
  else
    echo "[DONE] Plotted $count heatmaps from GSM8K dynamics files"
  fi
}

run_prompt() {
  if [ "$PROMPT_ENABLED" != "True" ] && [ "$PROMPT_ENABLED" != "true" ]; then
    echo "[SKIP] prompt.enabled=false"
    return
  fi

  echo "[RUN] Prompt generation chain"
  python3 generate_prompt.py \
    --model-id "$MODEL_ID" \
    --batch-size "$PROMPT_BATCH_SIZE" \
    --steps "$PROMPT_STEPS" \
    --gen-length "$PROMPT_GEN_LENGTH" \
    --block-length "$PROMPT_BLOCK_LENGTH" \
    --local-half-window "$PROMPT_LOCAL_HALF_WINDOW" \
    --temperature "$PROMPT_TEMPERATURE" \
    --cfg-scale "$PROMPT_CFG_SCALE" \
    --remasking "$PROMPT_REMASKING" \
    --results-dir "$RESULTS_DIR" \
    --device "$PROMPT_DEVICE"

  PROMPT_RESULTS_DIR="$RESULTS_DIR/prompt"
  mkdir -p "$PROMPT_RESULTS_DIR"
  count=0
  for npy in "$PROMPT_RESULTS_DIR"/prompt_dynamics_*.npy; do
    if [ ! -f "$npy" ]; then
      continue
    fi
    png="${npy%.npy}.png"
    title="$PROMPT_TITLE_PREFIX $(basename "$npy" .npy)"
    python3 plot_dynamics.py "$npy" --output "$png" --title "$title"
    count=$((count + 1))
  done

  if [ "$count" -eq 0 ]; then
    echo "[WARN] No Prompt dynamics npy files found in $PROMPT_RESULTS_DIR"
  else
    echo "[DONE] Plotted $count heatmaps from Prompt dynamics files"
  fi
}

run_longbench() {
  if [ "$LONGBENCH_ENABLED" != "True" ] && [ "$LONGBENCH_ENABLED" != "true" ]; then
    echo "[SKIP] longbench.enabled=false"
    return
  fi

  echo "[RUN] LongBench generation chain"
  python3 generate_longbench.py \
    --model-id "$MODEL_ID" \
    --datasets "$LONGBENCH_DATASETS" \
    --samples-per-dataset "$LONGBENCH_SAMPLES_PER_DATASET" \
    --batch-size "$LONGBENCH_BATCH_SIZE" \
    --steps "$LONGBENCH_STEPS" \
    --gen-length "$LONGBENCH_GEN_LENGTH" \
    --block-length "$LONGBENCH_BLOCK_LENGTH" \
    --local-half-window "$LONGBENCH_LOCAL_HALF_WINDOW" \
    --temperature "$LONGBENCH_TEMPERATURE" \
    --cfg-scale "$LONGBENCH_CFG_SCALE" \
    --remasking "$LONGBENCH_REMASKING" \
    --results-dir "$RESULTS_DIR" \
    --device "$LONGBENCH_DEVICE"

  LONGBENCH_RESULTS_DIR="$RESULTS_DIR/longbench"
  mkdir -p "$LONGBENCH_RESULTS_DIR"
  count=0
  for npy in "$LONGBENCH_RESULTS_DIR"/*_dynamics_*.npy; do
    if [ ! -f "$npy" ]; then
      continue
    fi
    png="${npy%.npy}.png"
    title="$LONGBENCH_TITLE_PREFIX $(basename "$npy" .npy)"
    python3 plot_dynamics.py "$npy" --output "$png" --title "$title"
    count=$((count + 1))
  done

  if [ "$count" -eq 0 ]; then
    echo "[WARN] No LongBench dynamics npy files found in $LONGBENCH_RESULTS_DIR"
  else
    echo "[DONE] Plotted $count heatmaps from LongBench dynamics files"
  fi
}


case "$MODE" in
  dummy)
    run_dummy
    ;;
  real)
    run_real
    ;;
  prompt)
    run_prompt
    ;;
  longbench)
    run_longbench
    ;;
  all)
    run_dummy
    run_real
    run_prompt
    run_longbench
    ;;
  *)
    echo "Usage: bash commands.sh [dummy|real|prompt|longbench|all] [config_path]"
    exit 1
    ;;
esac

echo "[DONE] Outputs in $RESULTS_DIR"
