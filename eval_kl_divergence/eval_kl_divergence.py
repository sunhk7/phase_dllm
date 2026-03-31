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
    """
    Load dataset, filter out strings that are too short,
    and return `num_samples` of tokenized inputs of exactly `max_length`.
    """
    print("Loading dataset PG-19 (streaming mode)...")
    try:
        # Use a pure parquet version of PG-19 that doesn't rely on deprecated Python scripts
        dataset = load_dataset("emozilla/pg19-test", split="train", streaming=True)
    except Exception:
        try:
            dataset = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        except Exception:
            # Absolute fallback
            dataset = load_dataset("deepmind/pg19", split="test", streaming=True, trust_remote_code=True)

    print(f"Filtering and tokenizing dataset for max_length={max_length}, num_samples={num_samples}...")
    valid_samples = []
    valid_texts = []
    
    # Shuffle buffer since it's a streaming dataset
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=100)
    
    for item in tqdm(shuffled_dataset, desc="Finding valid samples"):
        text = item.get('text', '')
        if not text.strip():
            continue
            
        # Fast heuristic: ensure the text is long enough without concatenation
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
        
    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_samples_maxlen_{max_length}.json", "w", encoding="utf-8") as f:
        json.dump(valid_texts, f, ensure_ascii=False, indent=2)
    print(f"Saved the extracted text samples to results/eval_samples_maxlen_{max_length}.json")
    
    return valid_samples

def evaluate_kl_divergence(model, valid_samples, local_window, device):
    """
    Performs dual forward pass mapping (Global vs target Local Window).
    Returns arrays suitable for CDF plotting.
    """
    kl_divergences = []

    print(f"Running evaluation for local_window={local_window}...")
    with torch.no_grad():
        for input_ids in tqdm(valid_samples, desc=f"Evaluating w={local_window}"):
            input_ids = input_ids.unsqueeze(0).to(device)
            
            # Global forward
            global_outputs = model(input_ids=input_ids, local_window_size=None)
            global_logits = global_outputs.logits if hasattr(global_outputs, 'logits') else global_outputs[0]
            global_probs = F.softmax(global_logits, dim=-1)
            
            # Local forward
            local_outputs = model(input_ids=input_ids, local_window_size=local_window)
            local_logits = local_outputs.logits if hasattr(local_outputs, 'logits') else local_outputs[0]
            local_probs = F.softmax(local_logits, dim=-1)
            
            # PyTorch F.kl_div(input, target) computes KL(target || input)
            # We want KL(local || global), so target is local_probs, and input is global_probs.log()
            kl = F.kl_div(global_probs.log(), local_probs, reduction='none').sum(dim=-1)
            kl = kl.squeeze(0).cpu().to(torch.float32).numpy() # shape: (max_length,)
            
            # Support tuple-based configuration (window_size, keep_first_k)
            if isinstance(local_window, tuple):
                w_size, keep_first_k = local_window
            else:
                w_size, keep_first_k = local_window, 0
            
            # We skip the KL divergence calculation for the prompt/sink tokens themselves, 
            # and we also skip the last `w_size` tokens due to chopped right-side windows.
            start_idx = max(w_size, keep_first_k)
            valid_kl = kl[start_idx:-w_size]
            kl_divergences.extend(valid_kl.tolist())

    print(f"Collected {len(kl_divergences)} valid KL divergence values for local_window={local_window}.")
    
    kl_divergences = np.array(kl_divergences)
    sorted_kl = np.sort(kl_divergences)
    
    # Calculate CDF
    p = 1.0 * np.arange(len(sorted_kl)) / (len(sorted_kl) - 1)
    
    os.makedirs("results", exist_ok=True)
    # create a safe filesystem string for file naming
    safe_name = str(local_window).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")
    with open(f"results/kl_divergences_window_{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(kl_divergences.tolist(), f)
    
    return sorted_kl, p

def plot_multiple_cdf(results_dict, max_length):
    """
    Takes a dictionary mapping string labels to (sorted_kl, p) tuples,
    and plots them on the same CDF chart for easy comparison.
    """
    plt.figure(figsize=(10, 6))
    
    for label, (sorted_kl, p) in results_dict.items():
        plt.plot(sorted_kl, p, label=label, linewidth=2)
        
    plt.xlim(0, 0.1)
    plt.ylim(0, 1.0)
    plt.axvline(x=0.01, color='red', linestyle='--', label='Threshold (0.01)')
    
    plt.title(f"CDF of KL Divergence between Local and Global Attention (maxlen={max_length})")
    plt.xlabel("KL Divergence")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    os.makedirs("results", exist_ok=True)
    out_path = f"results/kl_divergence_cdf_comparison_maxlen_{max_length}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison CDF plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    print(f"Loading local model using weights from {args.model_name_or_path}...")
    try:
        config = LLaDAConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(
            args.model_name_or_path, 
            config=config, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Failed to load with local class due to: {e}. Falling back to AutoModelForCausalLM.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        
    model = model.to(args.device)
    model.eval()

    # ==========================
    # EXPERIMENT CONFIGURATION
    # ==========================
    max_length = 1024
    num_samples = args.num_samples
    
    # NEW: To keep the first K tokens always visible (e.g., for System Prompts / Sink tokens),
    # pass a tuple like `(window_size, keep_first_k)`, for example `(16, 64)`.
    local_windows_to_test = [16, 32, 64, (16, 64)]
    # ==========================
    
    # 1. Fetch data samples (only needs to be done once per maximum sequence length)
    valid_samples = get_dataset_samples(tokenizer, max_length, num_samples)
    
    # 2. Extract kl divergence metrics for each specified window size configurations
    results_dict = {}
    for w in local_windows_to_test:
        sorted_kl, p = evaluate_kl_divergence(model, valid_samples, local_window=w, device=args.device)
        
        # Friendly label format
        if isinstance(w, tuple):
            label = f"Window = {w[0]}, Sink = {w[1]}"
        else:
            label = f"Window = {w}"
            
        results_dict[label] = (sorted_kl, p)
        
    # 3. Plot them all onto a single CDF graph
    plot_multiple_cdf(results_dict, max_length)

if __name__ == "__main__":
    main()
