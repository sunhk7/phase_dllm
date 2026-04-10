import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json
import os
import sys

from eval_kl_divergence import get_dataset_samples
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM

def compute_jsd_tensors(p, q):
    """ Computes Jensen-Shannon Divergence efficiently for stacked sequence tensors. """
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
    kl_qm = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def evaluate_shift_simulation(model, valid_samples, w_size, threshold, device):
    old_hard_jsds = []
    new_hard_jsds = []
    old_hard_matches = []
    new_hard_matches = []
    
    mask_id = 126336
    micro_batch_size = 2 
    eval_positions_per_seq = 64
    
    print(f"\n🚀 Running True Gt SHiFT Simulation for w_size={w_size} (Thresh < {threshold})")
    
    with torch.no_grad():
        for seq_idx, input_ids in enumerate(tqdm(valid_samples, desc="Simulating SHiFT")):
            input_ids = input_ids.unsqueeze(0).to(device)
            max_length = input_ids.shape[-1]
            
            # --- PHASE 1: Find Easy Tokens via Unmasked JS Divergence ---
            global_outputs = model(input_ids=input_ids, local_window_size=None)
            global_logits = global_outputs.logits if hasattr(global_outputs, 'logits') else global_outputs[0]
            global_probs = F.softmax(global_logits, dim=-1)
            
            local_outputs = model(input_ids=input_ids, local_window_size=w_size)
            local_logits = local_outputs.logits if hasattr(local_outputs, 'logits') else local_outputs[0]
            local_probs = F.softmax(local_logits, dim=-1)
            
            # JSD shape: (max_length,)
            unmasked_jsd = compute_jsd_tensors(global_probs, local_probs).squeeze(0)
            
            is_valid_range = torch.zeros_like(unmasked_jsd, dtype=torch.bool)
            is_valid_range[w_size:-w_size] = True
            
            is_easy = (unmasked_jsd < threshold) & is_valid_range
            is_hard = (unmasked_jsd >= threshold) & is_valid_range
            hard_indices = torch.nonzero(is_hard, as_tuple=True)[0].cpu().tolist()
            
            # 🔥 CRITICAL FIX: Kill Phase 1 giant variables before spinning up Phase 2!
            del global_outputs, global_logits, global_probs
            del local_outputs, local_logits, local_probs
            del unmasked_jsd
            torch.cuda.empty_cache()
            
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
                
                global_out = model(input_ids=batch_segments, local_window_size=None)
                global_probs_masked = F.softmax(global_out.logits if hasattr(global_out, 'logits') else global_out[0], dim=-1)
                
                old_local_out = model(input_ids=batch_segments, local_window_size=w_size)
                old_local_probs = F.softmax(old_local_out.logits if hasattr(old_local_out, 'logits') else old_local_out[0], dim=-1)
                
                new_local_out = model(input_ids=batch_segments, local_window_size=dynamic_window)
                new_local_probs = F.softmax(new_local_out.logits if hasattr(new_local_out, 'logits') else new_local_out[0], dim=-1)
                
                for i, relative_idx in enumerate(range(b_start, b_end)):
                    target_idx = chosen_hard_indices[relative_idx]
                    
                    p_g = global_probs_masked[i, target_idx]
                    p_old_loc = old_local_probs[i, target_idx]
                    p_new_loc = new_local_probs[i, target_idx]
                    
                    # Compute JS Divergence (mathematically symmetric & smoothed formulation of KL)
                    m_old = 0.5 * (p_g + p_old_loc)
                    kl_pm_old = F.kl_div(m_old.log(), p_g, reduction='sum')
                    kl_qm_old = F.kl_div(m_old.log(), p_old_loc, reduction='sum')
                    jsd_old = 0.5 * (kl_pm_old + kl_qm_old).item()
                    
                    m_new = 0.5 * (p_g + p_new_loc)
                    kl_pm_new = F.kl_div(m_new.log(), p_g, reduction='sum')
                    kl_qm_new = F.kl_div(m_new.log(), p_new_loc, reduction='sum')
                    jsd_new = 0.5 * (kl_pm_new + kl_qm_new).item()
                    
                    # Top-1 Accuracy exactly on the MASKED HARD tokens alone
                    old_acc = (p_g.argmax() == p_old_loc.argmax()).item()
                    new_acc = (p_g.argmax() == p_new_loc.argmax()).item()
                    
                    old_hard_jsds.append(jsd_old)
                    new_hard_jsds.append(jsd_new)
                    old_hard_matches.append(int(old_acc))
                    new_hard_matches.append(int(new_acc))
                    
    print(f"Collected paired Blind Tests for {len(old_hard_jsds)} unique Hard Tokens.")
    
    os.makedirs("results/eval_shift_simulation", exist_ok=True)
    # Save files identifying JS metric
    with open(f"results/eval_shift_simulation/shift_sim_jsd_w{w_size}_th{threshold}.json", "w") as f:
        json.dump({
            "old_jsds": old_hard_jsds,
            "new_jsds": new_hard_jsds,
            "old_matches": old_hard_matches,
            "new_matches": new_hard_matches
        }, f)
        
    return np.array(old_hard_jsds), np.array(new_hard_jsds)


def plot_shift_simulation(old_jsds, new_jsds, w_size, threshold):
    """ Plot JS Divergence CDF specifically. """
    plt.figure(figsize=(10, 6))
    
    old_sorted = np.sort(old_jsds)
    new_sorted = np.sort(new_jsds)
    
    p_old = 1.0 * np.arange(len(old_sorted)) / max(len(old_sorted) - 1, 1)
    p_new = 1.0 * np.arange(len(new_sorted)) / max(len(new_sorted) - 1, 1)
    
    plt.plot(old_sorted, p_old, label=f'Standard Window={w_size} (Hard Tokens)', linewidth=2, color='#E74C3C')
    plt.plot(new_sorted, p_new, label=f'SHiFT Window={w_size} + Global Gt (th={threshold})', linewidth=3, color='#2ECC71')
    
    # Range is dynamically contained typically [0, 0.1]. JSD tops theoretically at ln(2) ~0.69
    plt.xlim(0, 0.1)
    plt.ylim(0, 1.0)
    plt.axvline(x=0.01, color='gray', linestyle='--', label='JSD=0.01 Boundary')
    
    plt.title(f"SHiFT-dLLM Simulation: Masked JS Divergence (w_size={w_size}, th={threshold})")
    plt.xlabel("Jensen-Shannon Divergence Penalty (Blind Context Prediction)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    os.makedirs("results/eval_shift_simulation", exist_ok=True)
    out_path = f"results/eval_shift_simulation/shift_simulation_cdf_w{w_size}_th{threshold}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved breakthrough graph to \033[1;32m{out_path}\033[0m")


def plot_aggregated_top1():
    """ 
    Aggregates all JSON outputs globally to compute and visualize Top-1 acc 
    strictly on 'Hard Tokens' before and after introducing G_t landmarks across 8 GPUs. 
    """
    target_dir = "results/eval_shift_simulation"
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found. Computations not completed?")
        return

    json_files = [f for f in os.listdir(target_dir) if f.startswith("shift_sim_jsd_w") and f.endswith(".json")]
    if len(json_files) == 0:
        print("No evaluation JSON files found.")
        return
        
    labels = []
    old_accs = []
    new_accs = []
    
    parsed_files = []
    for f in json_files:
        # shift_sim_jsd_w16_th0.01.json
        parts = f.replace("shift_sim_jsd_w", "").replace(".json", "").split("_th")
        w_size = int(parts[0])
        th = float(parts[1])
        parsed_files.append((w_size, th, f))
        
    # Sort files to ensure logical X-axis flow (by threshold tightening, then window ascending)
    parsed_files.sort(key=lambda x: (-x[1], x[0])) 
    
    for w_size, th, f in parsed_files:
        with open(os.path.join(target_dir, f), "r") as root:
            data = json.load(root)
            old_m = np.mean(data["old_matches"]) * 100
            new_m = np.mean(data["new_matches"]) * 100
            
            labels.append(f"W={w_size}\nTh={th}")
            old_accs.append(old_m)
            new_accs.append(new_m)
            
    # 2x1 Bar Chart Layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharey=True)
    
    # Plot 1: Before G_t
    bars1 = ax1.bar(labels, old_accs, color='salmon', edgecolor='black')
    ax1.set_title("Top-1 Accuracy on MASKED HARD TOKENS BEFORE G_t (Standard Local Window Only)", fontsize=14, pad=10)
    ax1.set_ylabel("Agreement with Global (%)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 105)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Plot 2: After G_t
    bars2 = ax2.bar(labels, new_accs, color='mediumseagreen', edgecolor='black')
    ax2.set_title("Top-1 Accuracy on MASKED HARD TOKENS AFTER G_t (SHiFT: Window + Landmark Routing)", fontsize=14, pad=10)
    ax2.set_ylabel("Agreement with Global (%)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(pad=3.0)
    out_path = f"{target_dir}/aggregated_top1_agreement_2x1.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Aggregated Top-1 Accuracy Chart to \033[1;32m{out_path}\033[0m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--plot_only", action="store_true", help="Skip execution, solely aggregate Top-1 bar distribution maps")
    args = parser.parse_args()

    if args.plot_only:
        plot_aggregated_top1()
        return

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
    
    valid_samples = get_dataset_samples(tokenizer, max_length, args.num_samples, pos_id=0)
    
    old_jsds, new_jsds = evaluate_shift_simulation(model, valid_samples, w_size=args.window, threshold=args.threshold, device=args.device)
    
    plot_shift_simulation(old_jsds, new_jsds, w_size=args.window, threshold=args.threshold)

if __name__ == "__main__":
    main()
