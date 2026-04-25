import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_single_window_size(w, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 4 个对比项
    mode_keys = [
        ('baseline',             f'res_baseline_w{w}.json',              'Baseline'),
        ('local_window',         f'res_local_window_w{w}.json',          'Local Window'),
        ('swin_window',          f'res_swin_window_w{w}.json',           'Swin Window'),
        ('swin_window_compiled', f'res_swin_window_compiled_w{w}.json',  'Swin+Compile'),
    ]
    
    results = {}
    labels = []
    for key, fname, label in mode_keys:
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                results[key] = json.load(f)
            labels.append((key, label))
        else:
            print(f"Skipping {label}: {fpath} not found")
    
    if len(labels) < 2:
        print("Not enough data to plot.")
        return
    
    mode_names = [l[1] for l in labels]
    x = np.arange(len(mode_names))
    
    tps = [results[k]['tokens_per_sec'] for k, _ in labels]
    dec_tps = [results[k].get('decode_tokens_per_sec', 0) for k, _ in labels]
    latencies = [results[k].get('avg_latency', 0) for k, _ in labels]
    
    # 颜色：原版蓝色系, Swin+Compile 绿色
    colors_dec = ['#1f77b4', '#1f77b4', '#1f77b4', '#2ca02c'][:len(labels)]
    colors_e2e = ['#ff7f0e', '#ff7f0e', '#ff7f0e', '#98df8a'][:len(labels)]
    
    # ==========================
    # 1. Throughput & Latency
    # ==========================
    fig, ax1 = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    
    bars1 = ax1.bar(x - bar_width/2, dec_tps, bar_width, color=colors_dec, label='Decode Tokens/s')
    bars2 = ax1.bar(x + bar_width/2, tps, bar_width, color=colors_e2e, label='E2E Tokens/s')
    
    ax1.set_ylabel('Tokens / s')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_names)
    
    for bar in list(bars1) + list(bars2):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', 
                 va='bottom', ha='center', fontsize=9)
    
    ax2 = ax1.twinx()
    ax2.plot(x, latencies, color='#d62728', marker='o', linestyle='-', linewidth=2, markersize=8, label='E2E Latency (s)')
    ax2.set_ylabel('Total Generation Latency (s)')
    
    for i, lat in enumerate(latencies):
        ax2.annotate(f'{lat:.2f}s', (x[i], lat), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10, color='#d62728', fontweight='bold')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax1.set_title(f'Throughput & Latency Comparison (w={w})', pad=30)
    
    tps_path = os.path.join(save_dir, f'throughput_latency_w{w}.png')
    plt.savefig(tps_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {tps_path}")

    # ==========================
    # 2. Avg Step Latency Bar Chart
    # ==========================
    def get_step_lats(res):
        lats = res.get('latency_list', [])
        nfe = res.get('nfe', 1)
        if not lats or nfe == 0:
            return [0]
        return [(l / nfe) * 1000 for l in lats]
    
    all_step_lats = [get_step_lats(results[k]) for k, _ in labels]
    avg_step_lats = [np.mean(sl) for sl in all_step_lats]
    
    step_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_step_lats, 0.5, color=step_colors)
    ax.set_ylabel('Avg Single Step Latency (ms)')
    ax.set_title(f'Per-Step Decoding Latency (w={w})')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}ms', 
                va='bottom', ha='center', fontsize=10)
    
    step_lat_path = os.path.join(save_dir, f'step_latency_w{w}.png')
    plt.savefig(step_lat_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {step_lat_path}")
    
    # ==========================
    # 3. Box Plot for Jitter
    # ==========================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot([d if len(d) > 0 else [0] for d in all_step_lats], 
                    patch_artist=True, labels=mode_names)
    
    box_colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)]
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    ax.set_ylabel('Step Latency Distribution (ms)')
    ax.set_title(f'Latency Stability (w={w})')
    
    boxplot_path = os.path.join(save_dir, f'step_latency_boxplot_w{w}.png')
    plt.savefig(boxplot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {boxplot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    
    print(f"Generating plots for w={args.w} in {args.output_dir}...")
    plot_single_window_size(args.w, save_dir=args.output_dir)
