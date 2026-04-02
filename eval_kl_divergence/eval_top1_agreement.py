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
                
    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_samples_maxlen_{max_length}.json", "w", encoding="utf-8") as f:
        json.dump(valid_texts, f, ensure_ascii=False, indent=2)
    return valid_samples

def evaluate_top1_agreement(model, valid_samples, local_window, device):
    """
    Evaluates Top-1 token agreement between Global and Local configurations on completely unmasked text.
    """
    agreements = []
    
    # Support tuple-based configuration (window_size, keep_first_k)
    if isinstance(local_window, tuple):
        w_size, keep_first_k = local_window
    else:
        w_size, keep_first_k = local_window, 0
    
    # Sequence evaluation start idx
    start_idx = max(w_size, keep_first_k)

    print(f"Running Top-1 Agreement for local_window={local_window}...")
    with torch.no_grad():
        for input_ids in tqdm(valid_samples, desc=f"Evaluating w={local_window}"):
            input_ids = input_ids.unsqueeze(0).to(device)
            
            # Global forward
            global_outputs = model(input_ids=input_ids, local_window_size=None)
            global_logits = global_outputs.logits if hasattr(global_outputs, 'logits') else global_outputs[0]
            
            # Local forward
            local_outputs = model(input_ids=input_ids, local_window_size=local_window)
            local_logits = local_outputs.logits if hasattr(local_outputs, 'logits') else local_outputs[0]
            
            global_top1 = global_logits.argmax(dim=-1).squeeze(0)  # shape: (max_length,)
            local_top1 = local_logits.argmax(dim=-1).squeeze(0)    # shape: (max_length,)
            
            # Calculate match status array
            match_arr = (global_top1 == local_top1).cpu().numpy()
            
            # Skip invalid boundaries
            valid_matches = match_arr[start_idx:-w_size]
            agreements.extend(valid_matches.tolist())

    accuracy = np.mean(agreements) * 100.0  # Percentage
    print(f"Window={local_window} | Collected {len(agreements)} tokens | Top-1 Agreement: {accuracy:.4f}%")
    
    return accuracy

def plot_agreement_bar_chart(results_dict, max_length):
    plt.figure(figsize=(10, 6))
    
    labels = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    bars = plt.bar(labels, accuracies, color='skyblue', edgecolor='black')
    
    # Add text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0, 105) # Cap at 100 but give room for text
    plt.title(f"Top-1 Prediction Agreement (Local vs Global Attention)\nEvaluated on Unmasked Clean Text (maxlen={max_length})")
    plt.ylabel("Top-1 Match Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs("results", exist_ok=True)
    out_path = f"results/top1_agreement_comparison_maxlen_{max_length}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Top-1 Agreement Chart to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    try:
        config = LLaDAConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
    model = model.to(args.device)
    model.eval()

    max_length = 1024
    num_samples = args.num_samples
    local_windows_to_test = [16, 32, 64, (16, 64)]
    
    valid_samples = get_dataset_samples(tokenizer, max_length, num_samples)
    
    results_dict = {}
    for w in local_windows_to_test:
        label = f"Window {w[0]}, Sink {w[1]}" if isinstance(w, tuple) else f"Window {w}"
        acc = evaluate_top1_agreement(model, valid_samples, local_window=w, device=args.device)
        results_dict[label] = acc
        
    plot_agreement_bar_chart(results_dict, max_length)

if __name__ == "__main__":
    main()
