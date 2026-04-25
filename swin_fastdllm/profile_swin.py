import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.profiler

def run_profile():
    device = 'cuda:1'
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    model.model.config.local_window_size = 8
    model.model.config.shift_size = 4
    
    # 随便伪造长度为 1024 的文本，模拟当前 Block 的生成
    x = torch.randint(0, 32000, (1, 1024), device=device)
    
    modes = ['baseline', 'local_window', 'swin_window']
    out_file = "profiler_comparison.txt"
    
    with open(out_file, "w") as f:
        f.write("=== LLaDA Attention Mechanisms Profiling Report ===\n\n")
        
    for mode in modes:
        print(f"\n[{mode}] Warming up CUDA...")
        model.model.config.attention_mode = mode
        
        with torch.inference_mode():
            for _ in range(2):
                model(x)
                
        print(f"[{mode}] Profiling started...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA 
            ],
            record_shapes=True,
        ) as prof:
            with torch.inference_mode():
                model(x)
                
        table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        
        # 追加写入到文件
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
