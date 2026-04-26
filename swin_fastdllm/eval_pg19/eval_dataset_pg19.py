"""
PG19 数据集评估脚本：同时测量速度和质量。

质量指标：给定真实文本，mask 掉后半段，用模型去噪恢复，与 ground truth 对比计算 Top-1 Accuracy。
速度指标：tokens/s, 平均延迟。

Usage:
    python eval_dataset_pg19.py \
        --attention-mode swin_triton \
        --seq-len 512 --block-length 256 --w 32 \
        --num-samples 10 --output-dir results/eval_pg19/512/256/32
"""

import torch
import json
import os
import sys
import argparse

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache


def load_pg19_samples(tokenizer, seq_len, num_samples, mask_id=126336):
    """从 PG19 test split 加载 num_samples 条数据，每条截取 seq_len 个 token。"""
    ds = load_dataset("pg19", split="test", streaming=True)
    samples = []
    for item in ds:
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if len(ids) >= seq_len:
            samples.append(torch.tensor(ids[:seq_len], dtype=torch.long))
            if len(samples) >= num_samples:
                break
    return samples


def evaluate_sample(model, token_ids, gen_length, block_length, steps, mask_id=126336):
    """
    对单条样本做 mask-and-predict 评估。

    token_ids: [seq_len] 的完整 ground truth token 序列
    前半段作为 prompt，后 gen_length 个 token 被 mask 掉让模型恢复。

    Returns:
        accuracy: float (Top-1 准确率)
        tokens_per_sec: float
        latency: float (秒)
    """
    device = next(model.parameters()).device
    prompt_len = len(token_ids) - gen_length
    prompt = token_ids[:prompt_len].unsqueeze(0).to(device)          # [1, prompt_len]
    ground_truth = token_ids[prompt_len:prompt_len + gen_length].to(device)  # [gen_length]

    # 计时
    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    ev_start.record()
    out, nfe, p_time, d_time = generate_with_dual_cache(
        model, prompt, steps=steps, gen_length=gen_length, block_length=block_length
    )
    ev_end.record()
    torch.cuda.synchronize()

    latency = ev_start.elapsed_time(ev_end) / 1000.0  # 秒
    tokens_per_sec = gen_length / latency

    # 准确率
    generated = out[0, prompt_len:prompt_len + gen_length]
    correct = (generated == ground_truth).sum().item()
    accuracy = correct / gen_length

    return accuracy, tokens_per_sec, latency, nfe


def main():
    parser = argparse.ArgumentParser(description="PG19 Dataset Evaluation")
    parser.add_argument('--attention-mode', type=str, required=True,
                        choices=['baseline', 'local_window', 'swin_window', 'swin_triton'])
    parser.add_argument('--seq-len', type=int, default=512, help='Total sequence length (prompt + generation)')
    parser.add_argument('--block-length', type=int, default=256, help='Block length for generation')
    parser.add_argument('--w', type=int, default=32, help='Window size')
    parser.add_argument('--steps', type=int, default=128, help='Denoising steps')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of PG19 samples to evaluate')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--warmup', type=int, default=2, help='Warmup samples (not counted in results)')
    args = parser.parse_args()

    device = 'cuda'
    gen_length = args.block_length  # 生成长度 = block_length（单 block 评估）

    # 确保 seq_len > gen_length
    assert args.seq_len > gen_length, f"seq_len ({args.seq_len}) must be > block_length ({gen_length})"

    print(f"Mode: {args.attention_mode} | seq_len={args.seq_len} | block={args.block_length} | w={args.w}")
    print("Loading model...")
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', torch_dtype=torch.bfloat16
    ).to(device).eval()

    model.model.config.attention_mode = args.attention_mode
    model.model.config.local_window_size = args.w
    model.model.config.shift_size = args.w // 2

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    print("Loading PG19 samples...")
    total_needed = args.num_samples + args.warmup
    samples = load_pg19_samples(tokenizer, args.seq_len, total_needed)
    print(f"Loaded {len(samples)} samples (seq_len={args.seq_len})")

    # Warmup
    print(f"Warming up ({args.warmup} samples)...")
    for i in range(min(args.warmup, len(samples))):
        _ = evaluate_sample(model, samples[i], gen_length, args.block_length, args.steps)

    # Evaluate
    accs, tps_list, lat_list, nfe_list = [], [], [], []
    eval_samples = samples[args.warmup:]

    for i, token_ids in enumerate(eval_samples):
        acc, tps, lat, nfe = evaluate_sample(model, token_ids, gen_length, args.block_length, args.steps)
        accs.append(acc)
        tps_list.append(tps)
        lat_list.append(lat)
        nfe_list.append(nfe)
        print(f"  [{i+1}/{len(eval_samples)}] acc={acc:.4f} | tps={tps:.1f} | lat={lat:.2f}s")

    # Aggregate
    results = {
        'attention_mode': args.attention_mode,
        'seq_len': args.seq_len,
        'block_length': args.block_length,
        'window_size': args.w,
        'steps': args.steps,
        'num_samples': len(eval_samples),
        'avg_accuracy': sum(accs) / len(accs),
        'avg_tokens_per_sec': sum(tps_list) / len(tps_list),
        'avg_latency': sum(lat_list) / len(lat_list),
        'avg_nfe': sum(nfe_list) / len(nfe_list),
        'per_sample_accuracy': accs,
        'per_sample_tps': tps_list,
        'per_sample_latency': lat_list,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'eval_{args.attention_mode}_w{args.w}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"  Avg Accuracy: {results['avg_accuracy']:.4f}")
    print(f"  Avg TPS:      {results['avg_tokens_per_sec']:.1f}")
    print(f"  Avg Latency:  {results['avg_latency']:.2f}s")


if __name__ == '__main__':
    main()
