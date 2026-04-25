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
    except FileNotFoundError as e:
        print(f"Error: Required JSON files not found. Ensure inferences are complete.\n{e}")
        return

    modes = ['Baseline', 'Local Window', 'Swin Window']
    tps = [res_base['tokens_per_sec'], res_local['tokens_per_sec'], res_swin['tokens_per_sec']]
    dec_tps = [res_base.get('decode_tokens_per_sec', 0), res_local.get('decode_tokens_per_sec', 0), res_swin.get('decode_tokens_per_sec', 0)]
    latencies = [res_base.get('avg_latency', 0), res_local.get('avg_latency', 0), res_swin.get('avg_latency', 0)]
    
    x = np.arange(len(modes))
    
    # ==========================
    # 1. Plot Consolidated Throughput & Latency Chart
    # ==========================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    
    bars1 = ax1.bar(x - bar_width/2, dec_tps, bar_width, label='Decode Tokens/s', color='#1f77b4')
    bars2 = ax1.bar(x + bar_width/2, tps, bar_width, label='End-to-End Tokens/s', color='#ff7f0e')
    
    ax1.set_ylabel('Tokens / s')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=10)
        
    ax2 = ax1.twinx()
    line = ax2.plot(x, latencies, color='#d62728', marker='o', linestyle='-', linewidth=2, markersize=8, label='Total E2E Latency (s)')
    ax2.set_ylabel('Total Generation Latency (s)')
    
    for i, lat in enumerate(latencies):
        ax2.annotate(f'{lat:.2f} s', (x[i], lat), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11, color='#d62728', fontweight='bold')
        
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    ax1.set_title(f'Comprehensive Throughput & E2E Latency (w={w})', pad=30)
    
    tps_path = os.path.join(save_dir, f'throughput_latency_w{w}.png')
    plt.savefig(tps_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {tps_path}")

    # ==========================
    # 2. Plot Avg Step Latency (Bar Chart)
    # ==========================
    def get_step_lats(res):
        lats = res.get('latency_list', [])
        nfe = res.get('nfe', 1)
        if not lats or nfe == 0:
            return [0]
        return [(l / nfe) * 1000 for l in lats]
        
    step_lats_base = get_step_lats(res_base)
    step_lats_local = get_step_lats(res_local)
    step_lats_swin = get_step_lats(res_swin)
    
    avg_step_lats = [
        np.mean(step_lats_base),
        np.mean(step_lats_local),
        np.mean(step_lats_swin)
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, avg_step_lats, bar_width*1.5, color=['#d62728', '#1f77b4', '#2ca02c'])
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
    # 3. Plot Box Plot for Step Latency Jitter
    # ==========================
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [step_lats_base, step_lats_local, step_lats_swin]
    
    bp = ax.boxplot([d if len(d) > 0 else [0] for d in data], patch_artist=True, labels=modes)
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    
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
    parser.add_argument('--output-dir', type=str, default='results', help="Directory to load/save JSON tracking metrics")
    args = parser.parse_args()
    
    print(f"Generating isolated plots for Window Size = {args.w} in {args.output_dir}...")
    plot_single_window_size(args.w, save_dir=args.output_dir)
