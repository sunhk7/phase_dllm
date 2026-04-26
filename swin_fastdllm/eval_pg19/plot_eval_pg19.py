"""
PG19 评估结果可视化：双 Y 轴 (Accuracy + Tokens/s) 对比图 + 逐样本散点图。
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_eval_results(output_dir, w):
    mode_infos = [
        ('baseline',      '#1f77b4', 'Baseline'),
        ('local_window',  '#aec7e8', 'Local Window'),
        ('swin_window',   '#ff7f0e', 'Swin Window'),
        ('swin_triton',   '#d62728', 'Swin+Triton'),
    ]

    results = {}
    labels = []
    for key, color, label in mode_infos:
        fpath = os.path.join(output_dir, f'eval_{key}_w{w}.json')
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                results[key] = json.load(f)
            labels.append((key, color, label))
        else:
            print(f"Skipping {label}: {fpath} not found")

    if len(labels) < 2:
        print("Not enough data to plot.")
        return

    mode_names = [l[2] for l in labels]
    colors = [l[1] for l in labels]
    x = np.arange(len(mode_names))

    accs = [results[k]['avg_accuracy'] * 100 for k, _, _ in labels]
    tps = [results[k]['avg_tokens_per_sec'] for k, _, _ in labels]
    lats = [results[k]['avg_latency'] for k, _, _ in labels]

    seq_len = results[labels[0][0]]['seq_len']
    bl = results[labels[0][0]]['block_length']

    # =========================================
    # 1. Accuracy + TPS 双 Y 轴
    # =========================================
    fig, ax1 = plt.subplots(figsize=(12, 7))
    bar_w = 0.35

    bars_acc = ax1.bar(x - bar_w / 2, accs, bar_w, color=colors, alpha=0.85, label='Accuracy (%)')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_ylim(0, max(accs) * 1.3 if max(accs) > 0 else 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_names)

    for bar in bars_acc:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}%',
                 va='bottom', ha='center', fontsize=9)

    ax2 = ax1.twinx()
    bars_tps = ax2.bar(x + bar_w / 2, tps, bar_w, color=colors, alpha=0.4, label='Tokens/s')
    ax2.set_ylabel('Tokens / s')

    for bar in bars_tps:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}',
                 va='bottom', ha='center', fontsize=9, fontstyle='italic')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)
    ax1.set_title(f'PG19 Evaluation: Accuracy & Speed\n(seq_len={seq_len}, block={bl}, w={w})', pad=30)

    plt.savefig(os.path.join(output_dir, f'eval_accuracy_speed_w{w}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: eval_accuracy_speed_w{w}.png")

    # =========================================
    # 2. 逐样本 Accuracy 散点图
    # =========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, color, label in labels:
        per_acc = [a * 100 for a in results[key]['per_sample_accuracy']]
        ax.plot(range(len(per_acc)), per_acc, marker='o', color=color, label=label, alpha=0.8)

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(f'Per-Sample Accuracy on PG19 (seq_len={seq_len}, block={bl}, w={w})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, f'eval_per_sample_acc_w{w}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: eval_per_sample_acc_w{w}.png")

    # =========================================
    # 3. Latency 对比
    # =========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, lats, 0.5, color=colors)
    ax.set_ylabel('Avg Latency (s)')
    ax.set_title(f'PG19 Evaluation: Latency (seq_len={seq_len}, block={bl}, w={w})')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}s',
                va='bottom', ha='center', fontsize=10)

    plt.savefig(os.path.join(output_dir, f'eval_latency_w{w}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: eval_latency_w{w}.png")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  PG19 Evaluation Summary (seq={seq_len}, block={bl}, w={w})")
    print(f"{'='*60}")
    print(f"  {'Mode':<18} {'Accuracy':>10} {'TPS':>10} {'Latency':>10}")
    print(f"  {'-'*48}")
    for key, _, label in labels:
        r = results[key]
        print(f"  {label:<18} {r['avg_accuracy']*100:>9.2f}% {r['avg_tokens_per_sec']:>9.1f} {r['avg_latency']:>9.2f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--w', type=int, default=32)
    args = parser.parse_args()
    plot_eval_results(args.output_dir, args.w)
