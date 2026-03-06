#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"        # dummy | real | all
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

# dummy
emit('DUMMY_ENABLED', cfg['dummy']['enabled'])
emit('DUMMY_TOKENIZER', cfg['dummy']['tokenizer'])
emit('DUMMY_STEPS', cfg['dummy']['steps'])
emit('DUMMY_GEN_LENGTH', cfg['dummy']['gen_length'])
emit('DUMMY_BLOCK_LENGTH', cfg['dummy']['block_length'])
emit('DUMMY_DEVICE', cfg['dummy']['device'])
emit('DUMMY_NPY', cfg['dummy']['output_npy'])
emit('DUMMY_PNG', cfg['dummy']['output_png'])

# real
emit('REAL_ENABLED', cfg['real']['enabled'])
emit('REAL_SOURCE_NPY', cfg['real']['source_npy'])
emit('REAL_NPY', cfg['real']['output_npy'])
emit('REAL_PNG', cfg['real']['output_png'])
emit('REAL_TITLE', cfg['real']['heatmap_title'])
PY
)"

mkdir -p "$RESULTS_DIR"

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
  python3 generate.py

  if [ ! -f "$REAL_SOURCE_NPY" ]; then
    echo "[ERROR] Expected output not found: $REAL_SOURCE_NPY"
    exit 1
  fi

  mv -f "$REAL_SOURCE_NPY" "$REAL_NPY"
  python3 plot_dynamics.py "$REAL_NPY" --output "$REAL_PNG" --title "$REAL_TITLE"
}

case "$MODE" in
  dummy)
    run_dummy
    ;;
  real)
    run_real
    ;;
  all)
    run_dummy
    run_real
    ;;
  *)
    echo "Usage: bash commands.sh [dummy|real|all] [config_path]"
    exit 1
    ;;
esac

echo "[DONE] Outputs in $RESULTS_DIR"
