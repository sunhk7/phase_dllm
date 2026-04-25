import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_single_window_size(w, save_dir='results'):
    """
    Reads the results corresponding to window size `w`
    and plots the comparison among the three modes.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with open(f'{save_dir}/res_baseline_w{w}.json', 'r') as f:
            res_base = json.load(f)
        with open(f'{save_dir}/res_local_window_w{w}.json', 'r') as f:
            res_local = json.load(f)
        with open(f'{save_dir}/res_swin_window_w{w}.json', 'r') as f:
            res_swin = json.load(f)
        with open(f'{save_dir}/res_swin_window_pad_w{w}.json', 'r') as f:
            res_pad = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required JSON files not found. Ensure inferences are complete.\n{e}")
        return

    modes = ['Baseline', 'Local Window', 'Swin Window\n(Roll)', 'Swin Window\n(Pad)']
    tps = [res_base['tokens_per_sec'], res_local['tokens_per_sec'], res_swin['tokens_per_sec'], res_pad['tokens_per_sec']]
    mem = [res_base['max_mem'], res_local['max_mem'], res_swin['max_mem'], res_pad['max_mem']]
    
    x = np.arange(len(modes))
    width = 0.5
    
    # ==========================
    # 1. Plot Tokens / s
    # ==========================
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, tps, width, color=['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'])
    
    ax.set_ylabel('Tokens / s')
    ax.set_title(f'Throughput Comparison (w={w})')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    
    # Annotate bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=11)
        
    tps_path = os.path.join(save_dir, f'throughput_w{w}.png')
    plt.savefig(tps_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {tps_path}")

    # ==========================
    # 2. Plot Peak Max Memory
    # ==========================
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, mem, width, color=['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'])
    
    ax.set_ylabel('Peak VRAM Allocation (MB)')
    ax.set_title(f'VRAM Usage Comparison (w={w})')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    
    # Annotate bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.0f} MB', va='bottom', ha='center', fontsize=11)
        
    mem_path = os.path.join(save_dir, f'memory_w{w}.png')
    plt.savefig(mem_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {mem_path}")

    # ==========================
    # 3. Plot Avg Step Latency (Bar Chart)
    # ==========================
    def get_step_lats(res):
        lats = res.get('latency_list', [])
        nfe = res.get('nfe', 1)
        if not lats or nfe == 0:
            return [0]
        # convert total sequence latency to single step latency (ms/step)
        return [(l / nfe) * 1000 for l in lats]
        
    step_lats_base = get_step_lats(res_base)
    step_lats_local = get_step_lats(res_local)
    step_lats_swin = get_step_lats(res_swin)
    step_lats_pad = get_step_lats(res_pad)
    
    avg_step_lats = [
        np.mean(step_lats_base),
        np.mean(step_lats_local),
        np.mean(step_lats_swin),
        np.mean(step_lats_pad)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_step_lats, width, color=['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'])
    ax.set_ylabel('Avg Single Step Latency (ms)')
    ax.set_title(f'Average Per-Step Decoding Latency (w={w})')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f} ms', va='bottom', ha='center', fontsize=11)
        
    step_lat_path = os.path.join(save_dir, f'step_latency_w{w}.png')
    plt.savefig(step_lat_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {step_lat_path}")
    
    # ==========================
    # 4. Plot Box Plot for Step Latency Jitter
    # ==========================
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [step_lats_base, step_lats_local, step_lats_swin, step_lats_pad]
    
    # Filter out empty data from arrays completely if early runs errored
    bp = ax.boxplot([d if len(d) > 0 else [0] for d in data], patch_artist=True, labels=modes)
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
        
    ax.set_ylabel('Step Latency Distribution (ms)')
    ax.set_title(f'Per-Step Latency Stability & Jitter Spread (w={w})')
    
    boxplot_path = os.path.join(save_dir, f'step_latency_boxplot_w{w}.png')
    plt.savefig(boxplot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {boxplot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=8, help="Window size that was tested")
    args = parser.parse_args()
    
    print(f"Generating isolated plots for Window Size = {args.w}...")
    plot_single_window_size(args.w)
