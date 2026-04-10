# Attention and Distribution Evaluator Toolbox

This toolkit contains several precise scientific evaluation algorithms targeting the Local Sufficiency vs Global Attention tradeoff, designed specifically for investigating Diffusion / Masked Language Models (e.g. LLaDA). It strictly replicates analytical conditions (e.g. Leave-One-Out unbiasing, Top-1 distribution overlap) using high-throughput metrics caching.

## 📁 `model/modeling_llada.py`
- **What it does**: This is the underlying neural map. The critical injection occurs inside `_scaled_dot_product_attention`, where we bypassed native PyTorch block-mask handling to install explicit boolean tensor routing grids (`local_window_size` dictionary interception).
- **Core verification**: Enables the physical execution of true Shift-dLLM (Ground truth fixed tokens being fully broadcastable globally regardless of Local Temporal Distance).

---

## 📁 `eval_kl_divergence.py`
- **What it does**: Conducts an exhaustive batch-parallel Leave-One-Out (Token Masked) extraction simulation.
- **Core verification**: Validates what happens to the Kullback-Leibler (KL) Divergence penalty of a generic dataset token when substituting Global Self Attention uniformly for strict Sliding Window Attention ($w$). It quantifies mathematical signal disruption during purely "Blind Text Contextual Inference".
- **How to execute it independently**:
  ```bash
  python eval_kl_divergence.py --window 64
  ```

---

## 📁 `eval_shift_gt_simulation.py`  ⭐ The SHiFT Crown Jewel
- **What it does**: Dynamically executes the exact "Ground Truth Substitution" simulated condition.
  1. Identifies "Easy Tokens" ($KL < threshold$) utilizing a prior pass.
  2. Bakes those Easy Token spatial bounds unmasked unconditionally into the attention local sliding mask (representing `G_t`).
  3. Emits `masked` permutations strictly targeting the residual "Hard Tokens", tracking predictive collapse behavior across generic Sliding Windows vs Shift-G_t Supported bounds.
- **Core verification**: Mathematically validates the foundational theorem behind SHiFT: Exposing early convergence features collapses subsequent localized distribution deviation entirely, allowing infinite safe generation scaling. It graphs CDF mapping generated from Jensen-Shannon (JS) Divergence and Aggregates generic Top-1 bar bounds focusing critically on the Masked failure cases.
- **How to execute it independently**:
  ```bash
  python eval_shift_gt_simulation.py --window 32 --threshold 0.05
  ```

---

## 📁 `eval_top1_agreement.py`
- **What it does**: Evaluates pure categorical alignment accuracy on totally clean, standard contexts ($z_0$).
- **Core verification**: Serves as a theoretical bound validation asserting: if all token identities are fully legible, does dropping historical tokens impact Top-1 Argmax classification output (Hint: it drastically does not).
- **How to execute it**:
  ```bash
  python eval_top1_agreement.py
  ```

---

## 📁 `run_8gpu_shift_sweep.sh`
- **What it does**: Bash orchestrator mapping distinct ablation hyperparameter environments (Threshold thresholds $\times$ window intervals) cleanly into the memory streams of 8 distinct hardware GPUs, culminating in an integrated aggregation call (`--plot_only`) rendering unified matrices across distributed tasks.
- **How to execute it**:
  ```bash
  chmod +x run_8gpu_shift_sweep.sh
  ./run_8gpu_shift_sweep.sh
  ```
