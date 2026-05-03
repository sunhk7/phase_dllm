"""
PG19 数据集评估脚本：速度 + 质量（PPL + Accuracy）。

质量指标：
  1. PPL (Perplexity): 基于官方 get_log_likelihood 的蒙特卡洛估计，
     但通过 KV-cache + replace_position 路径走 swin attention，
     使不同 attention mode 的 PPL 有区分度。
  2. Top-1 Accuracy: 生成后与 ground truth 逐 token 对比（steps = gen_length）。

速度指标：tokens/s, 平均延迟。

Usage:
    python eval_dataset_pg19.py \
        --attention-mode swin_triton \
        --seq-len 512 --block-length 256 --w 32 \
        --num-samples 10 --output-dir results/eval_pg19/512/256/32
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
import argparse
import gc
import math

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pg19_samples(tokenizer, seq_len, num_samples):
    """从 PG19 test split 加载 num_samples 条数据，每条截取 seq_len 个 token。"""
    ds = load_dataset("deepmind/pg19", split="test", streaming=True, trust_remote_code=True)
    samples = []
    for item in ds:
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if len(ids) >= seq_len:
            samples.append(torch.tensor(ids[:seq_len], dtype=torch.long))
            if len(samples) >= num_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# PPL: Monte Carlo log-likelihood via cached forward pass
#
# Adapted from official LLaDA get_log_likelihood.py.
# Key difference: uses KV-cache + replace_position so that swin/local
# window attention is active during the "answer block" forward pass.
# ---------------------------------------------------------------------------

def forward_process_block(answer, mask_id, device):
    """
    Randomly mask a subset of answer tokens (single sample).
    Returns:
        masked_answer: [1, L_ans] with some tokens replaced by mask_id
        mask_index:    [1, L_ans] bool — True where masked
        mask_ratio:    float in (0, 1]
    """
    L = answer.shape[0]
    # Sample number of tokens to mask: k ∈ [1, L]
    k = torch.randint(1, L + 1, (), device=device).item()
    mask_ratio = k / L

    # Randomly choose k positions to mask
    perm = torch.randperm(L, device=device)
    mask_positions = perm[:k]

    masked_answer = answer.clone().unsqueeze(0).to(device)  # [1, L]
    masked_answer[0, mask_positions] = mask_id

    mask_index = torch.zeros(1, L, dtype=torch.bool, device=device)
    mask_index[0, mask_positions] = True

    return masked_answer, mask_index, mask_ratio


@torch.no_grad()
def compute_ppl_cached(model, token_ids, gen_length, mc_num=64, mask_id=126336):
    """
    Compute perplexity of the answer portion via Monte Carlo estimation,
    going through the KV-cache + replace_position path (swin attention active).

    Steps:
      1. Build full sequence [prompt | answer], prefill to get KV cache.
      2. For each MC iteration:
         a. Randomly mask some answer tokens.
         b. Forward the masked answer block through the cached path
            (replace_position activates swin attention).
         c. Compute cross_entropy(logits[masked], ground_truth[masked]) / mask_ratio.
      3. PPL = exp(avg_loss).

    Returns:
        ppl:      float
        neg_ll:   float (negative log-likelihood, lower is better)
    """
    device = next(model.parameters()).device
    prompt_len = len(token_ids) - gen_length
    prompt = token_ids[:prompt_len].to(device)
    answer = token_ids[prompt_len:prompt_len + gen_length].to(device)  # [gen_length]

    # --- Step 1: Build KV-cache on the FULL sequence (prompt + real answer) ---
    full_seq = token_ids.unsqueeze(0).to(device)  # [1, seq_len]
    out_full = model(full_seq, use_cache=True)
    base_past_kv = out_full.past_key_values

    # replace_position marks the answer block
    replace_position = torch.zeros(1, len(token_ids), dtype=torch.bool, device=device)
    s = prompt_len
    e = prompt_len + gen_length
    replace_position[:, s:e] = True

    # --- Step 2: Monte Carlo estimation ---
    losses = []
    for _ in range(mc_num):
        masked_answer, mask_index_blk, mask_ratio = forward_process_block(answer, mask_id, device)

        # Deep-clone the KV cache so each iteration starts from the same base
        past_kv = []
        for layer_k, layer_v in base_past_kv:
            past_kv.append((layer_k.clone(), layer_v.clone()))

        # Forward the masked answer block through the cached path
        logits_blk = model(
            masked_answer,
            past_key_values=past_kv,
            use_cache=True,
            replace_position=replace_position,
        ).logits  # [1, gen_length, vocab]

        # Cross-entropy on masked positions only
        masked_logits = logits_blk[mask_index_blk]        # [num_masked, vocab]
        masked_targets = answer.unsqueeze(0)[mask_index_blk]  # [num_masked]
        ce = F.cross_entropy(masked_logits.float(), masked_targets, reduction='none')

        # Importance-weight by 1/mask_ratio (ELBO estimator from LLaDA paper)
        loss = (ce / mask_ratio).sum().item()
        losses.append(loss)

    neg_ll = sum(losses) / len(losses)
    avg_per_token = neg_ll / gen_length
    ppl = math.exp(min(avg_per_token, 100))  # clamp to avoid inf

    return ppl, neg_ll


# ---------------------------------------------------------------------------
# Generation accuracy (with steps = gen_length, matching LLaDA paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_generation(model, token_ids, gen_length, block_length, steps, mask_id=126336):
    """
    Generate via diffusion denoising, compare with ground truth.

    Returns:
        accuracy: float (Top-1 token accuracy)
        tokens_per_sec: float
        latency: float (seconds)
        nfe: int
    """
    device = next(model.parameters()).device
    prompt_len = len(token_ids) - gen_length
    prompt = token_ids[:prompt_len].unsqueeze(0).to(device)
    ground_truth = token_ids[prompt_len:prompt_len + gen_length].to(device)

    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    ev_start.record()
    out, nfe, p_time, d_time = generate_with_dual_cache(
        model, prompt, steps=steps, gen_length=gen_length, block_length=block_length
    )
    ev_end.record()
    torch.cuda.synchronize()

    latency = ev_start.elapsed_time(ev_end) / 1000.0
    tokens_per_sec = gen_length / latency

    generated = out[0, prompt_len:prompt_len + gen_length]
    correct = (generated == ground_truth).sum().item()
    accuracy = correct / gen_length

    return accuracy, tokens_per_sec, latency, nfe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PG19 Dataset Evaluation (PPL + Accuracy)")
    parser.add_argument('--attention-mode', type=str, required=True,
                        choices=['baseline', 'local_window', 'swin_window', 'swin_triton'])
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Total sequence length (prompt + generation)')
    parser.add_argument('--block-length', type=int, default=256,
                        help='Block length for generation')
    parser.add_argument('--w', type=int, default=32, help='Window size')
    parser.add_argument('--steps', type=int, default=None,
                        help='Denoising steps (default: gen_length, matching LLaDA paper)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of PG19 samples to evaluate')
    parser.add_argument('--mc-num', type=int, default=64,
                        help='Monte Carlo iterations for PPL estimation')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--warmup', type=int, default=2,
                        help='Warmup samples (not counted in results)')
    parser.add_argument('--compile', action='store_true',
                        help='Wrap model with torch.compile for operator fusion')
    args = parser.parse_args()

    device = 'cuda'
    gen_length = args.block_length

    # Default: steps = gen_length (LLaDA paper: optimal when steps = response length)
    if args.steps is None:
        args.steps = gen_length

    assert args.seq_len > gen_length, \
        f"seq_len ({args.seq_len}) must be > block_length ({gen_length})"

    compiled_tag = '_compiled' if args.compile else ''
    print(f"Mode: {args.attention_mode}{compiled_tag} | seq_len={args.seq_len} "
          f"| block={args.block_length} | w={args.w} | steps={args.steps} | mc={args.mc_num}")

    # ---- Load model ----
    print("Loading model...")
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', torch_dtype=torch.bfloat16
    ).to(device).eval()

    model.model.config.attention_mode = args.attention_mode
    model.model.config.local_window_size = args.w
    model.model.config.shift_size = args.w // 2

    if args.compile:
        print("Applying torch.compile (mode=default)...")
        model = torch.compile(model, mode='default')

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # ---- Load data ----
    print("Loading PG19 samples...")
    total_needed = args.num_samples + args.warmup
    samples = load_pg19_samples(tokenizer, args.seq_len, total_needed)
    print(f"Loaded {len(samples)} samples (seq_len={args.seq_len})")

    # ---- Warmup ----
    print(f"Warming up ({args.warmup} samples)...")
    with torch.inference_mode():
        for i in range(min(args.warmup, len(samples))):
            _ = evaluate_generation(
                model, samples[i], gen_length, args.block_length, args.steps)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ---- Evaluate ----
    eval_samples = samples[args.warmup:]
    ppls, neg_lls = [], []
    accs, tps_list, lat_list, nfe_list = [], [], [], []

    for i, token_ids in enumerate(eval_samples):
        # 1) PPL via cached forward (swin attention active)
        ppl, neg_ll = compute_ppl_cached(
            model, token_ids, gen_length, mc_num=args.mc_num)
        ppls.append(ppl)
        neg_lls.append(neg_ll)

        # 2) Generation accuracy + speed
        acc, tps, lat, nfe = evaluate_generation(
            model, token_ids, gen_length, args.block_length, args.steps)
        accs.append(acc)
        tps_list.append(tps)
        lat_list.append(lat)
        nfe_list.append(nfe)

        print(f"  [{i+1}/{len(eval_samples)}] "
              f"ppl={ppl:.2f} | acc={acc:.4f} | tps={tps:.1f} | lat={lat:.2f}s")

    # ---- Aggregate ----
    results = {
        'attention_mode': args.attention_mode,
        'compiled': args.compile,
        'seq_len': args.seq_len,
        'block_length': args.block_length,
        'window_size': args.w,
        'steps': args.steps,
        'mc_num': args.mc_num,
        'num_samples': len(eval_samples),
        # Quality
        'avg_ppl': sum(ppls) / len(ppls),
        'avg_neg_ll': sum(neg_lls) / len(neg_lls),
        'avg_accuracy': sum(accs) / len(accs),
        # Speed
        'avg_tokens_per_sec': sum(tps_list) / len(tps_list),
        'avg_latency': sum(lat_list) / len(lat_list),
        'avg_nfe': sum(nfe_list) / len(nfe_list),
        # Per-sample
        'per_sample_ppl': ppls,
        'per_sample_accuracy': accs,
        'per_sample_tps': tps_list,
        'per_sample_latency': lat_list,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f'eval_{args.attention_mode}{compiled_tag}_w{args.w}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"  Avg PPL:      {results['avg_ppl']:.2f}")
    print(f"  Avg Accuracy: {results['avg_accuracy']:.4f}")
    print(f"  Avg TPS:      {results['avg_tokens_per_sec']:.1f}")
    print(f"  Avg Latency:  {results['avg_latency']:.2f}s")


if __name__ == '__main__':
    main()
