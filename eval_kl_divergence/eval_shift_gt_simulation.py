import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json
import os
import sys

# Import shared helpers from the previous kl script
from eval_kl_divergence import get_dataset_samples
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM


def evaluate_shift_simulation(model, valid_samples, w_size, threshold, device):
    """
    Simulates SHiFT-dLLM algorithm state by identifying early/easy tokens ("Landmarks"),
    exposing them globally via memory dicts, and strictly re-testing the Masked prediction 
    KL errors strictly on the remaining "Hard" tokens.
    """
    old_hard_kls = []
    new_hard_kls = []
    
    mask_id = 126336
    micro_batch_size = 4
    eval_positions_per_seq = 64 # Max hard tokens to actually test per sequence
    
    print(f"\n🚀 Running True Gt SHiFT Simulation for w_size={w_size} (Easy Landmark KL Threshold < {threshold})")
    
    with torch.no_grad():
        for seq_idx, input_ids in enumerate(tqdm(valid_samples, desc="Simulating SHiFT")):
            input_ids = input_ids.unsqueeze(0).to(device)
            max_length = input_ids.shape[-1]
            
            # --- PHASE 1: Find Easy Tokens (Landmarks) Using Unmasked Heuristic ---
            global_outputs = model(input_ids=input_ids, local_window_size=None)
            global_logits = global_outputs.logits if hasattr(global_outputs, 'logits') else global_outputs[0]
            global_probs = F.softmax(global_logits, dim=-1)
            
            local_outputs = model(input_ids=input_ids, local_window_size=w_size)
            local_logits = local_outputs.logits if hasattr(local_outputs, 'logits') else local_outputs[0]
            local_probs = F.softmax(local_logits, dim=-1)
            
            # (1, max_length) -> (max_length,)
            unmasked_kl = F.kl_div(global_probs.log(), local_probs, reduction='none').sum(dim=-1).squeeze(0)
            
            # Discard boundary edges from easy/hard taxonomy mapping 
            is_valid_range = torch.zeros_like(unmasked_kl, dtype=torch.bool)
            is_valid_range[w_size:-w_size] = True
            
            is_easy = (unmasked_kl < threshold) & is_valid_range
            is_hard = (unmasked_kl >= threshold) & is_valid_range
            
            hard_indices = torch.nonzero(is_hard, as_tuple=True)[0].cpu().tolist()
            
            if len(hard_indices) == 0:
                continue
                
            # --- PHASE 2: True Masked Retest Exclusively on Hard Tokens ---
            num_to_test = min(eval_positions_per_seq, len(hard_indices))
            np.random.seed(int(input_ids[0, 10].item()) + seq_idx) 
            chosen_hard_indices = np.random.choice(hard_indices, size=num_to_test, replace=False)
            
            dynamic_window = {
                'w_size': w_size,
                'global_mask': is_easy
            }
            
            masked_inputs = input_ids.repeat(num_to_test, 1)
            for b_idx, target_idx in enumerate(chosen_hard_indices):
                masked_inputs[b_idx, target_idx] = mask_id
            
            for b_start in range(0, num_to_test, micro_batch_size):
                b_end = min(b_start + micro_batch_size, num_to_test)
                batch_segments = masked_inputs[b_start:b_end]
                
                # 1. Global Gold Standard (Ground truth masked prediction)
                global_out = model(input_ids=batch_segments, local_window_size=None)
                global_logits_masked = global_out.logits if hasattr(global_out, 'logits') else global_out[0]
                global_probs_masked = F.softmax(global_logits_masked, dim=-1)
                
                # 2. Old Baseline (Standard w_size isolated context)
                old_local_out = model(input_ids=batch_segments, local_window_size=w_size)
                old_local_logits = old_local_out.logits if hasattr(old_local_out, 'logits') else old_local_out[0]
                old_local_probs = F.softmax(old_local_logits, dim=-1)
                
                # 3. Enhanced SHiFT Local (w_size + Gt Landmarks globally visible bounds)
                new_local_out = model(input_ids=batch_segments, local_window_size=dynamic_window)
                new_local_logits = new_local_out.logits if hasattr(new_local_out, 'logits') else new_local_out[0]
                new_local_probs = F.softmax(new_local_logits, dim=-1)
                
                for i, relative_idx in enumerate(range(b_start, b_end)):
                    target_idx = chosen_hard_indices[relative_idx]
                    
                    p_g = global_probs_masked[i, target_idx]
                    p_old_loc = old_local_probs[i, target_idx]
                    p_new_loc = new_local_probs[i, target_idx]
                    
                    kl_old = F.kl_div(p_g.log(), p_old_loc, reduction='sum').item()
                    kl_new = F.kl_div(p_g.log(), p_new_loc, reduction='sum').item()
                    
                    old_hard_kls.append(kl_old)
                    new_hard_kls.append(kl_new)
                    
    print(f"Collected paired Blind Tests for {len(old_hard_kls)} unique Hard Tokens.")
    
    os.makedirs("results/eval_shift_simulation", exist_ok=True)
    with open(f"results/eval_shift_simulation/shift_sim_kls_w{w_size}.json", "w") as f:
        json.dump({
            "old_kls": old_hard_kls,
            "new_kls": new_hard_kls
        }, f)
        
    return np.array(old_hard_kls), np.array(new_hard_kls)


def plot_shift_simulation(old_kls, new_kls, w_size):
    plt.figure(figsize=(10, 6))
    
    old_sorted = np.sort(old_kls)
    new_sorted = np.sort(new_kls)
    
    p_old = 1.0 * np.arange(len(old_sorted)) / max(len(old_sorted) - 1, 1)
    p_new = 1.0 * np.arange(len(new_sorted)) / max(len(new_sorted) - 1, 1)
    
    plt.plot(old_sorted, p_old, label=f'Standard Window={w_size}', linewidth=2, color='#E74C3C')
    plt.plot(new_sorted, p_new, label=f'SHiFT Window={w_size} + Global Gt Landmarks', linewidth=3, color='#2ECC71')
    
    plt.xlim(0, 0.1)
    plt.ylim(0, 1.0)
    plt.axvline(x=0.01, color='gray', linestyle='--', label='KL=0.01 Threshold')
    
    plt.title(f"SHiFT-dLLM Upper Bound Simulation: Masked KL Divergence on Hard Tokens")
    plt.xlabel("KL Divergence Penalty (Blind Context Prediction)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    os.makedirs("results/eval_shift_simulation", exist_ok=True)
    out_path = f"results/eval_shift_simulation/shift_simulation_cdf_w{w_size}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved breakthrough verification plot to \033[1;32m{out_path}\033[0m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--window", type=int, default=16, help="Standard minimum window size to test")
    parser.add_argument("--threshold", type=float, default=0.01, help="KL threshold to identify fixed landmark tokens")
    args = parser.parse_args()

    print(f"Loading Models & Tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    try:
        config = LLaDAConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
    model = model.to(args.device)
    model.eval()

    max_length = 1024
    
    # Extract robust dataset lines via shared helper function 
    valid_samples = get_dataset_samples(tokenizer, max_length, args.num_samples, pos_id=0)
    
    old_kls, new_kls = evaluate_shift_simulation(model, valid_samples, w_size=args.window, threshold=args.threshold, device=args.device)
    
    plot_shift_simulation(old_kls, new_kls, w_size=args.window)

if __name__ == "__main__":
    main()
