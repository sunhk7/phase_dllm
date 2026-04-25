import torch
import torch.profiler
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache

def run_profile():
    device = 'cuda:0'
    print("Loading model...")
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    model.model.config.local_window_size = 8
    model.model.config.shift_size = 4
    
    # 模拟真实前缀
    prompt = torch.randint(0, 32000, (1, 256), device=device)
    
    modes = ['baseline', 'local_window', 'swin_window']
    out_file = "profiler_comparison.txt"
    
    with open(out_file, "w") as f:
        f.write("=== LLaDA Attention Mechanisms Profiling Report ===\n\n")
        
    for mode in modes:
        print(f"\n[{mode}] Warming up CUDA...")
        model.model.config.attention_mode = mode
        
        # 预热一次短生成
        _ = generate_with_dual_cache(model, prompt, steps=4, gen_length=32, block_length=32)
                
        print(f"[{mode}] Profiling started...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA 
            ],
            record_shapes=True,
        ) as prof:
            # 正式评估：生成 1 个 Block，循环迭代 8 次
            _ = generate_with_dual_cache(model, prompt, steps=8, gen_length=32, block_length=32)
                
        table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        
        with open(out_file, "a") as f:
            f.write(f"===============================================================\n")
            f.write(f"                        MODE: {mode.upper()}\n")
            f.write(f"===============================================================\n")
            f.write(table_str)
            f.write("\n\n")
            
        print(f"[{mode}] Done! Results appended to {out_file}")
        
    print(f"\nAll profiling finished! Open '{out_file}' to compare the bottleneck operators.")

if __name__ == '__main__':
    run_profile()
