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

def get_dataset_samples(tokenizer, max_length, num_samples):
    print("Loading dataset PG-19 (streaming mode)...")
    try:
        dataset = load_dataset("emozilla/pg19-test", split="train", streaming=True)
    except Exception:
        try:
            dataset = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        except Exception:
            dataset = load_dataset("deepmind/pg19", split="test", streaming=True, trust_remote_code=True)

    print(f"Filtering and tokenizing dataset for max_length={max_length}, num_samples={num_samples}...")
    valid_samples = []
    valid_texts = []
    
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=100)
    
    for item in tqdm(shuffled_dataset, desc="Finding valid samples"):
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
                
    if len(valid_samples) < num_samples:
        print(f"Warning: Only found {len(valid_samples)} samples of length {max_length}")
        
    os.makedirs("results/eval_kl_divergence", exist_ok=True)
    with open(f"results/eval_kl_divergence/eval_samples_maxlen_{max_length}.json", "w", encoding="utf-8") as f:
        json.dump(valid_texts, f, ensure_ascii=False, indent=2)
    print(f"Saved the extracted text samples to results/eval_kl_divergence/eval_samples_maxlen_{max_length}.json")
    
    return valid_samples

def evaluate_kl_divergence(model, valid_samples, local_window, device):
    """
    Performs true "Leave-One-Out" masked forward pass mapping.
    """
    kl_divergences = []
    
    mask_id = 126336
    eval_positions_per_seq = 64
    micro_batch_size = 4
    
    print(f"Running REAL MASKED evaluation for local_window={local_window}...")
    
    if isinstance(local_window, tuple):
        w_size, keep_first_k = local_window
    else:
        w_size, keep_first_k = local_window, 0
        
    max_length = valid_samples[0].shape[-1]
    start_idx = max(w_size, keep_first_k)
    possible_indices = list(range(start_idx, max_length - w_size))
    
    with torch.no_grad():
        for input_ids in tqdm(valid_samples, desc=f"Evaluating w={local_window}"):
            np.random.seed(int(input_ids[10].item()))
            chosen_indices = np.random.choice(possible_indices, size=eval_positions_per_seq, replace=False)
            
            masked_inputs = input_ids.unsqueeze(0).repeat(eval_positions_per_seq, 1)
            for b_idx, target_idx in enumerate(chosen_indices):
                masked_inputs[b_idx, target_idx] = mask_id
            
            masked_inputs = masked_inputs.to(device)
            
            kl_for_this_seq = []
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
                    
                    kl = F.kl_div(p_global.log(), p_local, reduction='sum').item()
                    kl_for_this_seq.append(kl)
                    
            kl_divergences.extend(kl_for_this_seq)

    print(f"Collected {len(kl_divergences)} valid KL divergence values for local_window={local_window}.")
    
    os.makedirs("results/eval_kl_divergence", exist_ok=True)
    safe_name = str(local_window).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")
    with open(f"results/eval_kl_divergence/kl_divergences_window_{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(kl_divergences, f)
        
def plot_multiple_cdf(max_length=1024):
    """
    Finds all dumped jsons generated by parallel executions, and aggregates them into a final CDF graph.
    """
    target_dir = "results/eval_kl_divergence"
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found. Have you run the evaluations?")
        return

    json_files = [f for f in os.listdir(target_dir) if f.startswith("kl_divergences_window_") and f.endswith(".json")]
    if len(json_files) == 0:
        print("No evaluation JSON files found.")
        return
        
    plt.figure(figsize=(10, 6))
    
    for jfile in sorted(json_files):
        # Extract window config from filename
        config_str = jfile.replace("kl_divergences_window_", "").replace(".json", "")
        # Create a friendly label
        if "_" in config_str: # Tuple
            parts = config_str.split("_")
            label = f"Window {parts[0]}, Sink {parts[1]}"
        else:
            label = f"Window {config_str}"
            
        with open(os.path.join(target_dir, jfile), "r", encoding="utf-8") as f:
            kl_divergences = json.load(f)
            
        kl_divergences = np.array(kl_divergences)
        sorted_kl = np.sort(kl_divergences)
        
        # Calculate CDF
        p = 1.0 * np.arange(len(sorted_kl)) / max(len(sorted_kl) - 1, 1)
        
        plt.plot(sorted_kl, p, label=label, linewidth=2)
        
    plt.xlim(0, 0.1)
    plt.ylim(0, 1.0)
    plt.axvline(x=0.01, color='red', linestyle='--', label='Threshold (0.01)')
    
    plt.title(f"CDF of KL Divergence between Local and Global Attention (maxlen={max_length})")
    plt.xlabel("KL Divergence")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    out_path = f"{target_dir}/kl_divergence_cdf_comparison_maxlen_{max_length}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated comparison CDF plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--window", type=str, required=False, help="Window config, e.g. '16' or '16,64'")
    parser.add_argument("--plot_only", action="store_true", help="Skip evaluation and only aggregate json results into plot.")
    args = parser.parse_args()

    max_length = 1024
    
    if args.plot_only:
        print("Plot Only mode activated. Loading existing artifacts for aggregation...")
        plot_multiple_cdf(max_length=max_length)
        return
        
    if not args.window:
        print("Please provide a --window to evaluate, or invoke with --plot_only.")
        return

    # Parse window argument
    if "," in args.window:
        w = tuple(map(int, args.window.split(",")))
    else:
        w = int(args.window)

    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    print(f"Loading local model using weights from {args.model_name_or_path}...")
    try:
        config = LLaDAConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
    model = model.to(args.device)
    model.eval()
    
    # 1. Fetch data samples
    valid_samples = get_dataset_samples(tokenizer, max_length, args.num_samples)
    
    # 2. Extract kl divergence metrics ONLY for the specified configuration
    print(f"Starting isolated evaluation for Window={w} [Target Device: {args.device}]")
    evaluate_kl_divergence(model, valid_samples, local_window=w, device=args.device)
    print(f"Isolated evaluation for Window={w} completed.")

if __name__ == "__main__":
    main()
