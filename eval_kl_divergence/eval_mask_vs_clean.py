import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json
import os
import sys
import matplotlib.cm as cm

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

def evaluate_mask_vs_clean(model, valid_samples, w_config, mask_ratio, arg_flags, device, pos_id=0):
    mask_jsds = []
    clean_jsds = []
    mask_matches = []
    clean_matches = []
    
    mask_jsds_new = []
    mask_matches_new = []
    
    mask_id = 126336
    
    if isinstance(w_config, tuple):
        w_size, keep_first_k = w_config
        start_idx = max(w_size, keep_first_k)
        safe_w = f"{w_size}_{keep_first_k}"
    else:
        w_size = w_config
        keep_first_k = 0
        start_idx = w_size
        safe_w = str(w_size)
    
    with torch.no_grad():
        for seq_idx, input_ids in enumerate(tqdm(valid_samples, desc=f"GPU {pos_id} [Win: {str(w_config)}]", leave=True)):
            input_ids = input_ids.unsqueeze(0).to(device)
            max_length = input_ids.shape[-1]
            
            end_idx = max_length - w_size
            possible_indices = list(range(start_idx, end_idx))
            
            if len(possible_indices) <= 0:
                continue

            np.random.seed(int(input_ids[0, 10].item()) + seq_idx)
            num_to_mask = int(len(possible_indices) * mask_ratio)
            chosen_mask_indices = set(np.random.choice(possible_indices, size=num_to_mask, replace=False))
            
            masked_inputs = input_ids.clone()
            for idx in chosen_mask_indices:
                masked_inputs[0, idx] = mask_id

            # Global
            global_outputs = model(input_ids=masked_inputs, local_window_size=None)
            global_logits = global_outputs.logits if hasattr(global_outputs, 'logits') else global_outputs[0]
            global_probs = F.softmax(global_logits, dim=-1).squeeze(0) 
            
            # Old Local
            local_outputs = model(input_ids=masked_inputs, local_window_size=w_config)
            local_logits = local_outputs.logits if hasattr(local_outputs, 'logits') else local_outputs[0]
            local_probs = F.softmax(local_logits, dim=-1).squeeze(0) 

            jsd_old_all = compute_jsd_tensors(global_probs, local_probs) 
            global_argmax = global_probs.argmax(dim=-1)
            local_argmax = local_probs.argmax(dim=-1)
            matches_old_all = (global_argmax == local_argmax)

            # SHiFT New Local Integration
            jsd_new_all = None
            matches_new_all = None
            
            if arg_flags.use_shift:
                is_valid_range = torch.zeros_like(jsd_old_all, dtype=torch.bool)
                is_valid_range[start_idx:end_idx] = True
                
                # Dynamic routing using Phase 1 output dynamically mined from the current noise state $z_t$
                is_easy = (jsd_old_all < arg_flags.threshold) & is_valid_range
                
                dynamic_window = {
                    'w_size': w_size,
                    'keep_first_k': keep_first_k,
                    'global_mask': is_easy
                }
                
                new_local_outputs = model(input_ids=masked_inputs, local_window_size=dynamic_window)
                new_local_logits = new_local_outputs.logits if hasattr(new_local_outputs, 'logits') else new_local_outputs[0]
                new_local_probs = F.softmax(new_local_logits, dim=-1).squeeze(0)
                
                jsd_new_all = compute_jsd_tensors(global_probs, new_local_probs)
                new_local_argmax = new_local_probs.argmax(dim=-1)
                matches_new_all = (global_argmax == new_local_argmax)
                
                del new_local_outputs, new_local_logits, new_local_probs

            for idx in possible_indices:
                jsd = jsd_old_all[idx].item()
                match = int(matches_old_all[idx].item())
                if idx in chosen_mask_indices:
                    mask_jsds.append(jsd)
                    mask_matches.append(match)
                    if arg_flags.use_shift:
                        mask_jsds_new.append(jsd_new_all[idx].item())
                        mask_matches_new.append(int(matches_new_all[idx].item()))
                else:
                    clean_jsds.append(jsd)
                    clean_matches.append(match)
                    
            del global_outputs, global_logits, global_probs
            del local_outputs, local_logits, local_probs
            del jsd_old_all, matches_old_all, masked_inputs
            torch.cuda.empty_cache()
            
    target_dir = f"results/eval_mask_vs_clean/{mask_ratio}/{safe_w}_{mask_ratio}"
    os.makedirs(target_dir, exist_ok=True)
    out_json = f"{target_dir}/metrics.json"
    
    payload = {
        "mask_jsds": mask_jsds,
        "mask_matches": mask_matches,
        "clean_jsds": clean_jsds,
        "clean_matches": clean_matches,
        "w_label": f"[{safe_w.replace('_',',')}]"
    }
    
    if arg_flags.use_shift:
        payload["use_shift"] = True
        payload["threshold"] = arg_flags.threshold
        payload["mask_jsds_new"] = mask_jsds_new
        payload["mask_matches_new"] = mask_matches_new
        
    with open(out_json, "w") as f:
        json.dump(payload, f)


def plot_aggregated_results(condition, mask_ratio):
    """ Plots JS Divergence CDF and Top-1 for all found JSON metrics for a specific condition. """
    target_root = f"results/eval_mask_vs_clean/{mask_ratio}"
    if not os.path.exists(target_root):
        print(f"Directory {target_root} not found. Computations not complete?")
        return

    all_data = [] 
    
    for d in os.listdir(target_root):
        if d.endswith(f"_{mask_ratio}") and os.path.isdir(os.path.join(target_root, d)):
            dir_path = os.path.join(target_root, d)
            json_path = os.path.join(dir_path, "metrics.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as root:
                    data = json.load(root)
                    label = data.get("w_label", d.split(f"_{mask_ratio}")[0].replace('_', ','))
                    clean_label = label.replace('[', '').replace(']', '')
                    # Yields (16,) or (16, 64) depending on how many dims
                    sort_val = tuple(map(int, clean_label.split(',')))
                    
                    if condition == 'MASK':
                        jsds = np.array(data["mask_jsds"])
                        matches = np.array(data["mask_matches"])
                        acc = np.mean(matches) * 100 if len(matches) > 0 else 0
                        
                        if data.get("use_shift", False):
                            jsds_new = np.array(data["mask_jsds_new"])
                            matches_new = np.array(data["mask_matches_new"])
                            acc_new = np.mean(matches_new) * 100 if len(matches_new) > 0 else 0
                            
                            all_data.append(((sort_val, 0), f"{label} (Old)", jsds, acc, 'dashed'))
                            all_data.append(((sort_val, 1), f"{label} (SHiFT)", jsds_new, acc_new, 'solid'))
                        else:
                            all_data.append(((sort_val, 0), f"W={label}", jsds, acc, 'solid'))
                    else:
                        jsds = np.array(data["clean_jsds"])
                        matches = np.array(data["clean_matches"])
                        acc = np.mean(matches) * 100 if len(matches) > 0 else 0
                        all_data.append(((sort_val, 0), f"W={label}", jsds, acc, 'solid'))

    if not all_data:
        print(f"No JSON metrics found for ratio {mask_ratio} in {target_root}")
        return
            
    all_data.sort(key=lambda x: x[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Use distinct colors structurally depending on whether SHiFT mode is detected
    color_map = plt.cm.get_cmap('tab10', len(all_data)+1)
    
    for idx, (sort_val, label, jsds, acc, line_style) in enumerate(all_data):
        sorted_jsds = np.sort(jsds)
        p = 1.0 * np.arange(len(sorted_jsds)) / max(len(sorted_jsds) - 1, 1)
        
        c_idx = (idx // 2) if 'SHiFT' in label or 'Old' in label else idx
        
        ax1.plot(sorted_jsds, p, label=label, linewidth=3.0 if line_style=='solid' else 2.0, 
                 color=color_map(c_idx), linestyle=line_style, alpha=1.0 if line_style=='solid' else 0.7)
        
    ax1.set_xlim(0, 0.1)
    ax1.set_ylim(0, 1.0)
    ax1.axvline(x=0.01, color='gray', linestyle='--', label='JSD=0.01 Boundary')
    ax1.set_title(f"JS Divergence CDF for {condition} Tokens with mask_ratio {mask_ratio}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Jensen-Shannon Divergence Penalty")
    ax1.set_ylabel("Cumulative Probability")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    labels = [item[1] for item in all_data]
    accs = [item[3] for item in all_data]
    
    def get_bar_color(lbl):
        if 'SHiFT' in lbl: return '#2ECC71' # Green
        elif 'Old' in lbl: return '#E74C3C' # Red
        else: return 'mediumseagreen' if condition == 'MASK' else 'salmon'

    bar_colors = [get_bar_color(l) for l in labels]
    
    bars = ax2.bar(labels, accs, color=bar_colors, edgecolor='black', width=0.4)
    ax2.set_title(f"Top-1 Accuracy Agreement: {condition} Tokens", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Agreement with Global (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout(pad=3.0)
    
    shift_tag = "_shift" if any('SHiFT' in item[1] for item in all_data) else ""
    out_path = f"{target_root}/aggregated_{condition.lower()}_ratio{mask_ratio}{shift_tag}_2x1.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved aggregated {condition} graph to \033[1;32m{out_path}\033[0m")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--window", type=str, required=False, help="String window config eg '64' or '16,64'")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--pos_id", type=int, default=0)
    parser.add_argument("--use_shift", action="store_true", help="Enable generating Easy Token memory routing dynamically under noise")
    parser.add_argument("--threshold", type=float, default=0.01, help="JSD threshold determining Easy Tokens")
    args = parser.parse_args()

    max_length = 1024
    
    if args.plot_only:
        print("\nPlot Only mode activated. Aggregating all results...")
        plot_aggregated_results("MASK", args.mask_ratio)
        plot_aggregated_results("CLEAN", args.mask_ratio)
        return
        
    if not args.window:
        print("Please provide a --window to evaluate, or invoke with --plot_only.")
        return

    if "," in args.window:
        w_config = tuple(map(int, args.window.split(",")))
    else:
        w_config = int(args.window)

    if args.pos_id == 0:
        print(f"Loading Models & Tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    try:
        config = LLaDAConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
    model = model.to(args.device)
    model.eval()

    valid_samples = get_dataset_samples(tokenizer, max_length, args.num_samples, pos_id=args.pos_id)
    evaluate_mask_vs_clean(model, valid_samples, w_config, mask_ratio=args.mask_ratio, arg_flags=args, device=args.device, pos_id=args.pos_id)

if __name__ == "__main__":
    main()
