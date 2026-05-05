import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_single_window_size(w, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 6 个对比项
    mode_keys = [
        ('baseline',             f'res_baseline_w{w}.json',              'Baseline'),
        ('local_window',         f'res_local_window_w{w}.json',          'Local Window'),
        ('swin_window',          f'res_swin_window_w{w}.json',           'Swin Window'),
        ('swin_window_compiled', f'res_swin_window_compiled_w{w}.json',  'Swin+Compile'),
        ('swin_triton',          f'res_swin_triton_w{w}.json',           'Swin+Triton'),
        ('swin_triton_compiled', f'res_swin_triton_compiled_w{w}.json',  'Triton+Compile'),
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
    
    # 颜色：蓝=baseline系, 橙=swin, 绿=compile, 红=triton
    palette = {
        'baseline': '#1f77b4', 'local_window': '#aec7e8',
        'swin_window': '#ff7f0e', 'swin_window_compiled': '#2ca02c',
        'swin_triton': '#d62728', 'swin_triton_compiled': '#9467bd',
    }
    colors = [palette.get(k, '#999999') for k, _ in labels]
    
    # ==========================
    # 1. Throughput & Latency
    # ==========================
    fig, ax1 = plt.subplots(figsize=(13, 7))
    bar_width = 0.35
    
    bars1 = ax1.bar(x - bar_width/2, dec_tps, bar_width, color='#1f77b4', label='Decode Tokens/s', alpha=0.85)
    bars2 = ax1.bar(x + bar_width/2, tps, bar_width, color='#ff7f0e', label='E2E Tokens/s', alpha=0.85)
    
    ax1.set_ylabel('Tokens / s')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_names)
    
    for bar in list(bars1) + list(bars2):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', 
                 va='bottom', ha='center', fontsize=8)
    
    ax2 = ax1.twinx()
    ax2.plot(x, latencies, color='black', marker='o', linestyle='--', linewidth=2, markersize=8, label='E2E Latency (s)')
    ax2.set_ylabel('Total Latency (s)')
    for i, lat in enumerate(latencies):
        ax2.annotate(f'{lat:.2f}s', (x[i], lat), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', 
               bbox_to_anchor=(0.5, 1.05), ncol=3)
    ax1.set_title(f'Throughput & Latency (w={w})', pad=40)
    
    plt.savefig(os.path.join(save_dir, f'throughput_latency_w{w}.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # ==========================
    # 2. Per-Step Latency
    # ==========================
    def get_step_lats(res):
        lats = res.get('latency_list', [])
        nfe = res.get('nfe', 1)
        if not lats or nfe == 0: return [0]
        return [(l / nfe) * 1000 for l in lats]
    
    all_step_lats = [get_step_lats(results[k]) for k, _ in labels]
    avg_step_lats = [np.mean(sl) for sl in all_step_lats]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, avg_step_lats, 0.5, color=colors)
    ax.set_ylabel('Avg Step Latency (ms)')
    ax.set_title(f'Per-Step Decoding Latency (w={w})')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}ms', va='bottom', ha='center', fontsize=9)
    
    plt.savefig(os.path.join(save_dir, f'step_latency_w{w}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    # ==========================
    # 3. Box Plot
    # ==========================
    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot([d if len(d) > 0 else [0] for d in all_step_lats], 
                    patch_artist=True, labels=mode_names)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    ax.set_ylabel('Step Latency Distribution (ms)')
    ax.set_title(f'Latency Stability (w={w})')
    
    plt.savefig(os.path.join(save_dir, f'step_latency_boxplot_w{w}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved 3 plots to {save_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    plot_single_window_size(args.w, save_dir=args.output_dir)
