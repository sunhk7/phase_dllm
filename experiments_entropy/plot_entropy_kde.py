import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_entropy_kde(
    npy_path: str,
    output_path: str = "phase_transition_kde.png",
    title: str = "Entropy-Global Attention Phase Transition (2D KDE)",
) -> str:
    """读取 entropy-ratio 配对数据并绘制二维核密度图。"""
    pairs = np.load(npy_path)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"期望输入形状为 (N, 2)，实际得到 {pairs.shape}")

    # 第 0 列是 token 预测熵，第 1 列是全局注意力比例。
    entropy = pairs[:, 0]
    global_ratio = pairs[:, 1]

    # 删除无效值，避免 KDE 数值异常。
    valid = np.isfinite(entropy) & np.isfinite(global_ratio)
    entropy = entropy[valid]
    global_ratio = global_ratio[valid]
    if entropy.size == 0:
        raise RuntimeError("输入数据全部为 NaN/Inf，无法绘图。")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 8))

    # 使用二维核密度估计展示高密度区域，并打开 colorbar 体现密度强度。
    kde = sns.kdeplot(
        x=entropy,
        y=global_ratio,
        fill=True,
        cmap="mako",
        levels=60,
        thresh=0.01,
        cbar=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Token Prediction Entropy", fontsize=13)
    ax.set_ylabel("Global Attention Weight Ratio", fontsize=13)

    # 标注相变直觉区域：低熵更局部（Solid），高熵更全局（Liquid）。
    x_min, x_max = float(np.min(entropy)), float(np.max(entropy))
    y_min, y_max = float(np.min(global_ratio)), float(np.max(global_ratio))
    x_mid = x_min + 0.35 * (x_max - x_min)
    y_mid = y_min + 0.35 * (y_max - y_min)

    ax.text(
        x_min + 0.05 * (x_max - x_min),
        y_min + 0.10 * (y_max - y_min),
        "Solid Phase\n(Low Entropy, Local)",
        fontsize=11,
        color="white",
        bbox={"facecolor": "#0b1f2a", "alpha": 0.70, "pad": 6},
    )
    ax.text(
        x_mid,
        y_mid,
        "Liquid Phase\n(High Entropy, Global)",
        fontsize=11,
        color="white",
        bbox={"facecolor": "#1b4d5c", "alpha": 0.70, "pad": 6},
    )

    # seaborn.kdeplot(cbar=True) 会创建 colorbar，这里补充标签便于论文图引用。
    if kde.collections:
        cbar = kde.collections[0].colorbar
        if cbar is not None:
            cbar.set_label("Density", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot entropy/global-ratio KDE figure")
    parser.add_argument("npy_path", type=str, help="Path to entropy_ratio_pairs.npy")
    parser.add_argument("--output", type=str, default="phase_transition_kde.png")
    parser.add_argument("--title", type=str, default="Entropy-Global Attention Phase Transition (2D KDE)")
    args = parser.parse_args()

    output = plot_entropy_kde(args.npy_path, args.output, args.title)
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
