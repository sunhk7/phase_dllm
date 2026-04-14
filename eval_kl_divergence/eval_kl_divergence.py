import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM

# Using ANSI escape codes to position output elegantly in bash
def get_dataset_samples(tokenizer, max_length, num_samples, pos_id=0):
    # Hide initial print noise when running multiple at same time
    if pos_id == 0:
        print("Loading dataset PG-19 (streaming mode)...")
        
    try:
        dataset = load_dataset("emozilla/pg19-test", split="train", streaming=True)
    except Exception:
        try:
            dataset = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        except Exception:
            dataset = load_dataset("deepmind/pg19", split="test", streaming=True, trust_remote_code=True)

    valid_samples = []
    valid_texts = []
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=100)
    
    for item in tqdm(shuffled_dataset, desc=f"Loading data (GPU {pos_id})", position=pos_id, leave=True, mininterval=1.0):
        text = item.get('text', '')
        if not text.strip():
            continue
            
        if len(text) < max_length * 3:
            continue
            
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokens.input_ids[0]
        
        if len(input_ids) == max_length:
            valid_samples.append(input_ids)
            valid_texts.append(tokenizer.decode(input_ids, skip_special_tokens=True))
            
            if len(valid_samples) == num_samples:
                break
                
    if pos_id == 0:
        os.makedirs("results/eval_kl_divergence", exist_ok=True)
        with open(f"results/eval_kl_divergence/eval_samples_maxlen_{max_length}.json", "w", encoding="utf-8") as f:
            json.dump(valid_texts, f, ensure_ascii=False, indent=2)
    
    return valid_samples

def evaluate_divergences(model, valid_samples, local_window, device, pos_id=0):
    kl_divergences = []
    js_divergences = []
    
    mask_id = 126336
    eval_positions_per_seq = 64
    micro_batch_size = 4
    
    if isinstance(local_window, tuple):
        w_size, keep_first_k = local_window
    else:
        w_size, keep_first_k = local_window, 0
        
    max_length = valid_samples[0].shape[-1]
    start_idx = max(w_size, keep_first_k)
    possible_indices = list(range(start_idx, max_length - w_size))
    
    with torch.no_grad():
        for input_ids in tqdm(valid_samples, desc=f"GPU {pos_id} [Win: {str(local_window):<8}]", position=pos_id, leave=True):
            np.random.seed(int(input_ids[10].item()))
            chosen_indices = np.random.choice(possible_indices, size=eval_positions_per_seq, replace=False)
            
            masked_inputs = input_ids.unsqueeze(0).repeat(eval_positions_per_seq, 1)
            for b_idx, target_idx in enumerate(chosen_indices):
                masked_inputs[b_idx, target_idx] = mask_id
            
            masked_inputs = masked_inputs.to(device)
            
            kl_for_this_seq = []
            js_for_this_seq = []
            
            for b_start in range(0, eval_positions_per_seq, micro_batch_size):
                b_end = min(b_start + micro_batch_size, eval_positions_per_seq)
                batch_segments = masked_inputs[b_start:b_end]
                
                global_outputs = model(input_ids=batch_segments, local_window_size=None)
                global_logits = global_outputs.logits if hasattr(global_outputs, 'logits') else global_outputs[0]
                global_probs = F.softmax(global_logits, dim=-1)
                
                local_outputs = model(input_ids=batch_segments, local_window_size=local_window)
                local_logits = local_outputs.logits if hasattr(local_outputs, 'logits') else local_outputs[0]
                local_probs = F.softmax(local_logits, dim=-1)
                
                for i, relative_idx in enumerate(range(b_start, b_end)):
                    target_idx = chosen_indices[relative_idx]
                    p_global = global_probs[i, target_idx]
                    p_local = local_probs[i, target_idx]
                    
                    # 1. KL Divergence (Asymmetrical)
                    kl = F.kl_div(p_global.log(), p_local, reduction='sum').item()
                    
                    # 2. JS Divergence (Symmetrical & Smoothed)
                    m = 0.5 * (p_global + p_local)
                    kl_pm = F.kl_div(m.log(), p_global, reduction='sum')
                    kl_qm = F.kl_div(m.log(), p_local, reduction='sum')
                    js = 0.5 * (kl_pm + kl_qm).item()
                    
                    kl_for_this_seq.append(kl)
                    js_for_this_seq.append(js)
                    
            kl_divergences.extend(kl_for_this_seq)
            js_divergences.extend(js_for_this_seq)
    
    # Save isolated arrays to structurally separated subdirectories
    os.makedirs("results/eval_kl_divergence/kl", exist_ok=True)
    os.makedirs("results/eval_kl_divergence/js", exist_ok=True)
    
    safe_name = str(local_window).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")
    
    with open(f"results/eval_kl_divergence/kl/kl_divergences_window_{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(kl_divergences, f)
        
    with open(f"results/eval_kl_divergence/js/js_divergences_window_{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(js_divergences, f)
        
def plot_metric_cdf(metric="kl", max_length=1024):
    target_dir = f"results/eval_kl_divergence/{metric}"
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found. Have you executed the parallel evaluation sweeps?")
        return

    prefix = f"{metric}_divergences_window_"
    json_files = [f for f in os.listdir(target_dir) if f.startswith(prefix) and f.endswith(".json")]
    
    if len(json_files) == 0:
        print(f"No {metric.upper()} JSON files found in {target_dir}")
        return
        
    plt.figure(figsize=(10, 6))
    
    for jfile in sorted(json_files):
        config_str = jfile.replace(prefix, "").replace(".json", "")
        if "_" in config_str:
            parts = config_str.split("_")
            label = f"Window {parts[0]}, Sink {parts[1]}"
        else:
            label = f"Window {config_str}"
            
        with open(os.path.join(target_dir, jfile), "r", encoding="utf-8") as f:
            divergences = json.load(f)
            
        divergences = np.array(divergences)
        sorted_div = np.sort(divergences)
        
        p = 1.0 * np.arange(len(sorted_div)) / max(len(sorted_div) - 1, 1)
        plt.plot(sorted_div, p, label=label, linewidth=2)
        
    plt.xlim(0, 0.1)
    plt.ylim(0, 1.0)
    
    if metric == "kl":
        plt.axvline(x=0.01, color='red', linestyle='--', label='KL=0.01 Threshold')
        plt.title(f"CDF of KL Divergence between Local and Global Attention (maxlen={max_length})")
        plt.xlabel("KL Divergence Penalty")
        out_path = f"results/eval_kl_divergence/{metric}/kl_divergence_cdf_comparison_maxlen_{max_length}.png"
    else:
        plt.axvline(x=0.01, color='red', linestyle='--', label='JSD=0.01 Threshold')
        plt.title(f"CDF of Jensen-Shannon Divergence between Local and Global (maxlen={max_length})")
        plt.xlabel("Jensen-Shannon Divergence Penalty")
        out_path = f"results/eval_kl_divergence/{metric}/js_divergence_cdf_comparison_maxlen_{max_length}.png"
        
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated {metric.upper()} comparison CDF plot to \033[1;32m{out_path}\033[0m")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--window", type=str, required=False, help="Window config, e.g. '16' or '16,64'")
    parser.add_argument("--plot_only", action="store_true", help="Skip evaluation and only aggregate json results.")
    parser.add_argument("--pos_id", type=int, default=0, help="Vertical position index for terminal tqdm display")
    args = parser.parse_args()

    max_length = 1024
    
    if args.plot_only:
        print("\nPlot Only mode activated. Aggregating all KL and JS structural directories...")
        plot_metric_cdf(metric="kl", max_length=max_length)
        plot_metric_cdf(metric="js", max_length=max_length)
        return
        
    if not args.window:
        print("Please provide a --window to evaluate, or invoke with --plot_only.")
        return

    if "," in args.window:
        w = tuple(map(int, args.window.split(",")))
    else:
        w = int(args.window)

    if args.pos_id == 0:
        print(f"Loading Models from {args.model_name_or_path}...")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    try:
        config = LLaDAConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
    model = model.to(args.device)
    model.eval()
    
    valid_samples = get_dataset_samples(tokenizer, max_length, args.num_samples, pos_id=args.pos_id)
    
    evaluate_divergences(model, valid_samples, local_window=w, device=args.device, pos_id=args.pos_id)

if __name__ == "__main__":
    main()
