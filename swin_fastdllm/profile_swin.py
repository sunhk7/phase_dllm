import torch
import torch.profiler
import argparse
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache

def run_profile():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--w', type=int, default=32)
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

    # 4 个对比项：baseline, local_window, swin_window, swin_window+compile
    configs = [
        ('baseline',             'baseline',     model),
        ('local_window',         'local_window', model),
        ('swin_window',          'swin_window',  model),
        ('swin_window_compiled', 'swin_window',  compiled_model),
    ]

    out_file = f"profiler_comparison_bl{block_length}.txt"
    with open(out_file, "w") as f:
        f.write(f"=== Profiling Report | block_length={block_length}, w={w}, steps={steps} ===\n\n")

    for label, attn_mode, mdl in configs:
        is_compiled = 'compiled' in label
        warmup_rounds = 5 if is_compiled else 3

        print(f"\n[{label}] Warming up ({warmup_rounds} rounds)...")
        model.model.config.attention_mode = attn_mode

        for _ in range(warmup_rounds):
            _ = generate_with_dual_cache(mdl, prompt, steps=4,
                                         gen_length=block_length, block_length=block_length)

        print(f"[{label}] Profiling...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
        ) as prof:
            _ = generate_with_dual_cache(mdl, prompt, steps=steps,
                                         gen_length=block_length, block_length=block_length)

        table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)

        with open(out_file, "a") as f:
            f.write(f"{'='*70}\n")
            f.write(f"  MODE: {label.upper()} | block_length={block_length}\n")
            f.write(f"{'='*70}\n")
            f.write(table_str)
            f.write("\n\n")

        print(f"[{label}] Done!")

    print(f"\nAll done! Results saved to '{out_file}'")

if __name__ == '__main__':
    run_profile()
