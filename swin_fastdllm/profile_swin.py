"""
增强版 Profiler：同时输出端到端延迟 + 算子级分解。

每个模式输出：
  1. 端到端延迟（CUDA Event 计时，3 次取中位数）
  2. Prefill / Decode 分阶段耗时
  3. 算子级 Top-30 热点表（torch.profiler）

Usage:
    python profile_swin.py --block-length 256 --w 32 --steps 8
"""
import torch
import torch.profiler
import argparse
import gc
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache


def measure_latency(mdl, prompt, steps, block_length, repeat=5):
    """用 CUDA Event 测端到端延迟，返回中位数。"""
    lats, p_times, d_times = [], [], []
    with torch.inference_mode():
        for _ in range(repeat):
            torch.cuda.synchronize()
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            _, nfe, p_t, d_t = generate_with_dual_cache(
                mdl, prompt, steps=steps,
                gen_length=block_length, block_length=block_length
            )
            ev_e.record()
            torch.cuda.synchronize()
            lats.append(ev_s.elapsed_time(ev_e))  # ms
            p_times.append(p_t * 1000)  # → ms
            d_times.append(d_t * 1000)
    lats.sort()
    p_times.sort()
    d_times.sort()
    mid = len(lats) // 2
    return lats[mid], p_times[mid], d_times[mid], nfe


def run_profile():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=7, help='Latency measurement repeats (median used)')
    args = parser.parse_args()

    block_length = args.block_length
    steps = args.steps
    w = args.w
    S = w // 2

    device = 'cuda:0'
    print(f"Config: block_length={block_length}, steps={steps}, w={w}, S={S}")
    print("Loading model...")

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    model.model.config.local_window_size = w
    model.model.config.shift_size = S

    print("Compiling model...")
    compiled_model = torch.compile(model, mode='default')

    prompt = torch.randint(0, 32000, (1, 256), device=device)

    configs = [
        ('baseline',             'baseline',     model),
        ('local_window',         'local_window', model),
        ('swin_window',          'swin_window',  model),
        ('swin_window_compiled', 'swin_window',  compiled_model),
        ('swin_triton',          'swin_triton',  model),
    ]

    out_file = f"profiler_comparison_bl{block_length}.txt"
    with open(out_file, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  Profiling Report | block_length={block_length}, w={w}, steps={steps}\n")
        f.write(f"{'='*70}\n\n")

    # Summary table for quick comparison
    summary_lines = []
    summary_lines.append(f"{'Mode':<25} {'Total(ms)':>10} {'Prefill(ms)':>12} {'Decode(ms)':>11} {'TPS':>8}")
    summary_lines.append("-" * 70)

    for label, attn_mode, mdl in configs:
        print(f"\n{'='*50}")
        print(f"  [{label}]")
        print(f"{'='*50}")

        # ---- Clean GPU state between modes ----
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        model.model.config.attention_mode = attn_mode

        # ---- Warmup ----
        print(f"  Warming up ({args.warmup} rounds)...")
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = generate_with_dual_cache(mdl, prompt, steps=4,
                                             gen_length=block_length, block_length=block_length)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ---- 1. End-to-end latency (CUDA Event, median) ----
        print(f"  Measuring latency ({args.repeat} runs, median)...")
        total_ms, prefill_ms, decode_ms, nfe = measure_latency(
            mdl, prompt, steps, block_length, repeat=args.repeat
        )
        tps = block_length / (total_ms / 1000.0)

        summary_lines.append(f"{label:<25} {total_ms:>10.1f} {prefill_ms:>12.1f} {decode_ms:>11.1f} {tps:>8.1f}")

        # ---- 2. Operator-level profiling ----
        print(f"  Profiling operators...")
        gc.collect()
        torch.cuda.empty_cache()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
        ) as prof:
            with torch.inference_mode():
                _ = generate_with_dual_cache(mdl, prompt, steps=steps,
                                             gen_length=block_length, block_length=block_length)

        table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)

        # ---- Write to file ----
        with open(out_file, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"  MODE: {label.upper()}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"  End-to-End Latency (median of {args.repeat} runs):\n")
            f.write(f"    Total:   {total_ms:.1f} ms\n")
            f.write(f"    Prefill: {prefill_ms:.1f} ms\n")
            f.write(f"    Decode:  {decode_ms:.1f} ms\n")
            f.write(f"    NFE:     {nfe}\n")
            f.write(f"    TPS:     {tps:.1f} tokens/s\n\n")
            f.write(f"  Operator Breakdown (Top 30 by CUDA time):\n")
            f.write(table_str)
            f.write("\n")

        print(f"  Done! Total={total_ms:.1f}ms, TPS={tps:.1f}")

    # ---- Write summary table ----
    summary_text = "\n".join(summary_lines)
    with open(out_file, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"  SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(summary_text)
        f.write("\n")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(summary_text)
    print(f"\nFull report saved to '{out_file}'")


if __name__ == '__main__':
    run_profile()
