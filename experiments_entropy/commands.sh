#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-wikitext}"      # wikitext | all
CONFIG_PATH="${2:-config.yaml}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] config not found: $CONFIG_PATH"
  exit 1
fi

# 从 config.yaml 中读取关键参数（依赖 PyYAML）
eval "$(python3 - "$CONFIG_PATH" <<'PY'
import shlex
import sys

try:
    import yaml
except Exception:
    raise SystemExit("[ERROR] PyYAML is required: pip install pyyaml")

cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))

def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")

emit('SEED', cfg['experiment']['seed'])
emit('RESULTS_DIR', cfg['paths']['results_dir'])
emit('DATASET_SUBDIR', cfg['paths']['dataset_subdir'])
emit('PAIRS_NAME_SUFFIX', cfg['paths']['pairs_name_suffix'])
emit('FIGURE_NAME_SUFFIX', cfg['paths']['figure_name_suffix'])
emit('JSONL_FILENAME', cfg['paths']['jsonl_filename'])
emit('DATASET_NAME', cfg['data']['dataset_name'])
emit('DATASET_SPLIT', cfg['data']['dataset_split'])
emit('SAMPLES', cfg['data']['samples'])
emit('TOKENIZER_ID', cfg['data']['tokenizer_id'])
emit('PROMPT_LENGTH', cfg['data']['prompt_length'])
emit('GEN_LENGTH', cfg['data']['gen_length'])
emit('STEPS', cfg['data']['steps'])
emit('BLOCK_LENGTH', cfg['data']['block_length'])
emit('DEVICE', cfg['runtime']['device'])
emit('TITLE', cfg['plot']['title'])
emit('RUN_WIKITEXT', cfg['run']['wikitext'])
PY
)"

WIKITEXT_RESULTS_DIR="${RESULTS_DIR}/${DATASET_SUBDIR}"
OUTPUT_JSONL="${WIKITEXT_RESULTS_DIR}/${JSONL_FILENAME}"

mkdir -p "$WIKITEXT_RESULTS_DIR"

run_wikitext() {
  if [ "$RUN_WIKITEXT" != "True" ] && [ "$RUN_WIKITEXT" != "true" ]; then
    echo "[SKIP] run.wikitext=false"
    return
  fi

  echo "[RUN] Wikitext collect + plot"
  python3 collect_entropy_data.py \
    --dataset-name "$DATASET_NAME" \
    --dataset-split "$DATASET_SPLIT" \
    --samples "$SAMPLES" \
    --tokenizer-id "$TOKENIZER_ID" \
    --prompt-length "$PROMPT_LENGTH" \
    --gen-length "$GEN_LENGTH" \
    --steps "$STEPS" \
    --block-length "$BLOCK_LENGTH" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --output-dir "$WIKITEXT_RESULTS_DIR" \
    --pairs-name-suffix "$PAIRS_NAME_SUFFIX" \
    --output-jsonl "$OUTPUT_JSONL"

  SAFE_DATASET="$(python3 - <<'PY' "$DATASET_NAME"
import re
import sys
name = sys.argv[1]
safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.-")
print(safe or "dataset")
PY
)"

  echo "[RUN] Plot phase-transition KDEs for ${SAMPLES} samples"
  plotted=0
  i=0
  while [ "$i" -lt "$SAMPLES" ]; do
    sample_id=$(printf "%03d" "$i")
    npy_path="${WIKITEXT_RESULTS_DIR}/${SAFE_DATASET}_sample_${sample_id}_${PAIRS_NAME_SUFFIX}.npy"
    png_path="${WIKITEXT_RESULTS_DIR}/${SAFE_DATASET}_sample_${sample_id}_${FIGURE_NAME_SUFFIX}.png"

    if [ ! -f "$npy_path" ]; then
      echo "[ERROR] Missing sample npy: $npy_path"
      exit 1
    fi

    python3 plot_entropy_kde.py "$npy_path" \
      --output "$png_path" \
      --title "${TITLE} (${DATASET_NAME} sample ${sample_id})"
    plotted=$((plotted + 1))
    i=$((i + 1))
  done
  echo "[DONE] Generated ${plotted} sample PNGs in ${WIKITEXT_RESULTS_DIR}"
}

case "$MODE" in
  wikitext)
    run_wikitext
    ;;
  all)
    run_wikitext
    ;;
  *)
    echo "Usage: bash commands.sh [wikitext|all] [config.yaml]"
    exit 1
    ;;
esac
