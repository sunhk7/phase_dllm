#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"          # wikitext | prompt | gsm8k | all
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

# 从 config.yaml 中读取关键参数（依赖 PyYAML）
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

emit('SEED', get('experiment.seed', 42))
emit('MODEL_SOURCE', get('model.source', 'pretrained'))
emit('MODEL_ID', get('model.model_id', 'GSAI-ML/LLaDA-8B-Instruct'))
emit('TRUST_REMOTE_CODE', get('model.trust_remote_code', True))
emit('MODEL_DTYPE', get('model.dtype', 'auto'))
emit('RESULTS_DIR', get('paths.results_dir', 'results'))
emit('WIKITEXT_SUBDIR', get('paths.wikitext_subdir', 'wikitext'))
emit('PROMPT_SUBDIR', get('paths.prompt_subdir', 'prompt'))
emit('GSM8K_SUBDIR', get('paths.gsm8k_subdir', 'gsm8k'))
emit('PAIRS_NAME_SUFFIX', get('paths.pairs_name_suffix', 'entropy_ratio_pairs'))
emit('FIGURE_NAME_SUFFIX', get('paths.figure_name_suffix', 'phase_transition_hist'))
emit('JSONL_FILENAME', get('paths.jsonl_filename', 'wikitext_outputs.jsonl'))

emit('DATASET_NAME', get('data.dataset_name', 'wikitext-103-v1'))
emit('DATASET_SPLIT', get('data.dataset_split', 'test'))
emit('TASK_NAME', get('data.task_name', 'text_continuation'))
emit('SAMPLES', get('data.samples', 5))
emit('TOKENIZER_ID', get('data.tokenizer_id', 'GSAI-ML/LLaDA-8B-Instruct'))
emit('PROMPT_LENGTH', get('data.prompt_length', 256))
emit('GEN_LENGTH', get('data.gen_length', 768))
emit('STEPS', get('data.steps', 64))
emit('BLOCK_LENGTH', get('data.block_length', 768))

emit('WIKITEXT_ENABLED', get('wikitext.enabled', True))
emit('PROMPT_ENABLED', get('prompt.enabled', False))
emit('PROMPTS_FILE', get('prompt.prompts_file', 'prompts/prompts.txt'))
emit('PROMPT_KEY', get('prompt.prompt_key', 'prompt'))
emit('PROMPT_STRATEGIES', get('prompt.strategies', 'A,B,C'))
emit('PROMPT_ENTROPY_QUANTILE', get('prompt.entropy_quantile', 0.5))
emit('PROMPT_BATCH_SIZE', get('prompt.batch_size', 1))
emit('PROMPT_STEPS', get('prompt.steps', 64))
emit('PROMPT_GEN_LENGTH', get('prompt.gen_length', 768))
emit('PROMPT_BLOCK_LENGTH', get('prompt.block_length', 768))
emit('PROMPT_TEMPERATURE', get('prompt.temperature', 0.0))
emit('PROMPT_CFG_SCALE', get('prompt.cfg_scale', 0.0))
emit('PROMPT_REMASKING', get('prompt.remasking', 'low_confidence'))

emit('GSM8K_ENABLED', get('gsm8k.enabled', False))
emit('GSM8K_SPLIT', get('gsm8k.split', 'test'))
emit('GSM8K_MAX_SAMPLES', get('gsm8k.max_samples', 100))
emit('GSM8K_BATCH_SIZE', get('gsm8k.batch_size', 1))
emit('GSM8K_STEPS', get('gsm8k.steps', 64))
emit('GSM8K_GEN_LENGTH', get('gsm8k.gen_length', 768))
emit('GSM8K_BLOCK_LENGTH', get('gsm8k.block_length', 768))
emit('GSM8K_TEMPERATURE', get('gsm8k.temperature', 0.0))
emit('GSM8K_CFG_SCALE', get('gsm8k.cfg_scale', 0.0))
emit('GSM8K_REMASKING', get('gsm8k.remasking', 'low_confidence'))

emit('DEVICE', get('runtime.device', 'auto'))
emit('WIKITEXT_TITLE', get('plot.wikitext_title', 'Entropy-Global Attention Phase Transition (2D Histogram)'))
emit('PROMPT_TITLE_PREFIX', get('plot.prompt_title_prefix', 'Prompt Entropy-Global Histogram'))
emit('GSM8K_TITLE_PREFIX', get('plot.gsm8k_title_prefix', 'GSM8K Entropy-Global Histogram'))

emit('RUN_WIKITEXT', get('run.wikitext', True))
emit('RUN_PROMPT', get('run.prompt', False))
emit('RUN_GSM8K', get('run.gsm8k', False))
PY
)"

WIKITEXT_RESULTS_DIR="${RESULTS_DIR}/${WIKITEXT_SUBDIR}"
PROMPT_RESULTS_DIR="${RESULTS_DIR}/${PROMPT_SUBDIR}"
GSM8K_RESULTS_DIR="${RESULTS_DIR}/${GSM8K_SUBDIR}"
OUTPUT_JSONL="${WIKITEXT_RESULTS_DIR}/${JSONL_FILENAME}"

mkdir -p "$WIKITEXT_RESULTS_DIR"
mkdir -p "$PROMPT_RESULTS_DIR"
mkdir -p "$GSM8K_RESULTS_DIR"

run_wikitext() {
  if [ "$RUN_WIKITEXT" != "True" ] && [ "$RUN_WIKITEXT" != "true" ]; then
    echo "[SKIP] run.wikitext=false"
    return
  fi
  if [ "$WIKITEXT_ENABLED" != "True" ] && [ "$WIKITEXT_ENABLED" != "true" ]; then
    echo "[SKIP] wikitext.enabled=false"
    return
  fi

  echo "[RUN] Wikitext collect + plot"
  python3 collect_entropy_data.py \
    --model-source "$MODEL_SOURCE" \
    --model-id "$MODEL_ID" \
    --trust-remote-code "$TRUST_REMOTE_CODE" \
    --model-dtype "$MODEL_DTYPE" \
    --dataset-name "$DATASET_NAME" \
    --dataset-split "$DATASET_SPLIT" \
    --task-name "$TASK_NAME" \
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

  echo "[RUN] Plot wikitext histograms for ${SAMPLES} samples"
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
      --title "${WIKITEXT_TITLE} (${DATASET_NAME} sample ${sample_id})"
    plotted=$((plotted + 1))
    i=$((i + 1))
  done
  echo "[DONE] Generated ${plotted} wikitext sample PNGs in ${WIKITEXT_RESULTS_DIR}"
}

run_prompt() {
  if [ "$RUN_PROMPT" != "True" ] && [ "$RUN_PROMPT" != "true" ]; then
    echo "[SKIP] run.prompt=false"
    return
  fi
  if [ "$PROMPT_ENABLED" != "True" ] && [ "$PROMPT_ENABLED" != "true" ]; then
    echo "[SKIP] prompt.enabled=false"
    return
  fi

  echo "[RUN] Prompt generation"
  if [ -f "$PROMPTS_FILE" ]; then
    python3 generate_prompt.py \
      --model-id "$MODEL_ID" \
      --prompts-file "$PROMPTS_FILE" \
      --prompt-key "$PROMPT_KEY" \
      --strategies "$PROMPT_STRATEGIES" \
      --entropy-quantile "$PROMPT_ENTROPY_QUANTILE" \
      --batch-size "$PROMPT_BATCH_SIZE" \
      --steps "$PROMPT_STEPS" \
      --gen-length "$PROMPT_GEN_LENGTH" \
      --block-length "$PROMPT_BLOCK_LENGTH" \
      --temperature "$PROMPT_TEMPERATURE" \
      --cfg-scale "$PROMPT_CFG_SCALE" \
      --remasking "$PROMPT_REMASKING" \
      --results-dir "$RESULTS_DIR" \
      --device "$DEVICE" \
      "${EXTRA_ARGS[@]}"
  else
    echo "[WARN] prompts file not found: $PROMPTS_FILE, fallback to DEFAULT_PROMPTS"
    python3 generate_prompt.py \
      --model-id "$MODEL_ID" \
      --strategies "$PROMPT_STRATEGIES" \
      --entropy-quantile "$PROMPT_ENTROPY_QUANTILE" \
      --batch-size "$PROMPT_BATCH_SIZE" \
      --steps "$PROMPT_STEPS" \
      --gen-length "$PROMPT_GEN_LENGTH" \
      --block-length "$PROMPT_BLOCK_LENGTH" \
      --temperature "$PROMPT_TEMPERATURE" \
      --cfg-scale "$PROMPT_CFG_SCALE" \
      --remasking "$PROMPT_REMASKING" \
      --results-dir "$RESULTS_DIR" \
      --device "$DEVICE" \
      "${EXTRA_ARGS[@]}"
  fi

  count=0
  for npy in "$PROMPT_RESULTS_DIR"/prompt_*_pairs_*.npy; do
    [ -f "$npy" ] || continue
    png="${npy%.npy}_${FIGURE_NAME_SUFFIX}.png"
    title="${PROMPT_TITLE_PREFIX} $(basename "$npy" .npy)"

    meta="${npy/_pairs_/_meta_}"
    plot_input="$npy"
    plot_extra=()
    if [ -f "$meta" ]; then
      plot_input="$meta"
      plot_extra=(--only-updated)
      title="${title} (updated-only)"
    fi

    python3 plot_entropy_kde.py "$plot_input" --output "$png" --title "$title" "${plot_extra[@]}"
    count=$((count + 1))
  done
  if [ "$count" -eq 0 ]; then
    echo "[WARN] No prompt pair npy files found in $PROMPT_RESULTS_DIR"
  else
    echo "[DONE] Generated $count prompt PNGs in $PROMPT_RESULTS_DIR"
  fi
}

run_gsm8k() {
  if [ "$RUN_GSM8K" != "True" ] && [ "$RUN_GSM8K" != "true" ]; then
    echo "[SKIP] run.gsm8k=false"
    return
  fi
  if [ "$GSM8K_ENABLED" != "True" ] && [ "$GSM8K_ENABLED" != "true" ]; then
    echo "[SKIP] gsm8k.enabled=false"
    return
  fi

  echo "[RUN] GSM8K generation"
  python3 generate_gsm8k.py \
    --model-id "$MODEL_ID" \
    --split "$GSM8K_SPLIT" \
    --max-samples "$GSM8K_MAX_SAMPLES" \
    --batch-size "$GSM8K_BATCH_SIZE" \
    --steps "$GSM8K_STEPS" \
    --gen-length "$GSM8K_GEN_LENGTH" \
    --block-length "$GSM8K_BLOCK_LENGTH" \
    --temperature "$GSM8K_TEMPERATURE" \
    --cfg-scale "$GSM8K_CFG_SCALE" \
    --remasking "$GSM8K_REMASKING" \
    --results-dir "$RESULTS_DIR" \
    --device "$DEVICE"

  count=0
  for npy in "$GSM8K_RESULTS_DIR"/gsm8k_pairs_*.npy; do
    [ -f "$npy" ] || continue
    png="${npy%.npy}_${FIGURE_NAME_SUFFIX}.png"
    title="${GSM8K_TITLE_PREFIX} $(basename "$npy" .npy)"
    python3 plot_entropy_kde.py "$npy" --output "$png" --title "$title"
    count=$((count + 1))
  done
  if [ "$count" -eq 0 ]; then
    echo "[WARN] No gsm8k pair npy files found in $GSM8K_RESULTS_DIR"
  else
    echo "[DONE] Generated $count GSM8K PNGs in $GSM8K_RESULTS_DIR"
  fi
}

case "$MODE" in
  wikitext)
    run_wikitext
    ;;
  prompt)
    run_prompt
    ;;
  gsm8k)
    run_gsm8k
    ;;
  all)
    run_wikitext
    run_prompt
    run_gsm8k
    ;;
  *)
    echo "Usage: bash commands.sh [wikitext|prompt|gsm8k|all] [config.yaml] [extra args...]"
    exit 1
    ;;
esac
