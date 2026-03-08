import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _binned_mean_curve(x: np.ndarray, y: np.ndarray, bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    edges = np.linspace(x_min, x_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    idx = np.digitize(x, edges) - 1

    mean_y = np.full((bins,), np.nan, dtype=np.float32)
    for i in range(bins):
        mask = idx == i
        if np.any(mask):
            mean_y[i] = float(np.mean(y[mask]))

    valid = np.isfinite(mean_y)
    return centers[valid], mean_y[valid]


def plot_entropy_kde(
    npy_path: str,
    output_path: str = "phase_transition_kde.png",
    title: str = "Entropy vs Global Attention Ratio",
    mode: str = "both",
    only_updated: bool = False,
) -> str:
    """读取 entropy-ratio 配对数据并绘制散点/二维直方图（不做 KDE 平滑）。"""
    pairs = np.load(npy_path)
    if pairs.ndim != 2 or pairs.shape[1] < 2:
        raise ValueError(f"期望输入形状为 (N, >=2)，实际得到 {pairs.shape}")

    # 第 0 列是 token 预测熵，第 1 列是全局注意力比例。
    entropy = pairs[:, 0]
    global_ratio = pairs[:, 1]

    # 如果输入是 meta（>=3 列），可只保留“本步被更新”的 token，避免重复统计固化 token。
    if only_updated and pairs.shape[1] >= 3:
        updated = pairs[:, 2]
        updated_mask = np.isfinite(updated) & (updated > 0.5)
    else:
        updated_mask = np.ones_like(entropy, dtype=bool)

    # 删除无效值，避免二维直方图数值异常。
    valid = np.isfinite(entropy) & np.isfinite(global_ratio) & updated_mask
    entropy = entropy[valid]
    global_ratio = global_ratio[valid]
    if entropy.size == 0:
        raise RuntimeError("输入数据全部为 NaN/Inf，无法绘图。")

    mode = mode.lower()
    if mode not in {"scatter", "hist", "both"}:
        raise ValueError(f"Unknown mode: {mode}, choose from scatter|hist|both")

    sns.set_theme(style="whitegrid", context="talk")
    if mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        scatter_ax, hist_ax = axes
    elif mode == "scatter":
        fig, scatter_ax = plt.subplots(1, 1, figsize=(10, 7))
        hist_ax = None
    else:
        fig, hist_ax = plt.subplots(1, 1, figsize=(10, 7))
        scatter_ax = None

    x_curve, y_curve = _binned_mean_curve(entropy, global_ratio, bins=30)

    if scatter_ax is not None:
        # 用稀疏 alpha 散点展示真实点云形状。
        scatter_ax.scatter(entropy, global_ratio, s=2, alpha=0.08, c="#0b7285", edgecolors="none")
        if x_curve.size > 0:
            scatter_ax.plot(x_curve, y_curve, color="#d9480f", linewidth=2.0, label="Binned mean")
            scatter_ax.legend(loc="best", frameon=True)
        scatter_ax.set_title("Scatter", fontsize=14)
        scatter_ax.set_xlabel("Token Prediction Entropy", fontsize=12)
        scatter_ax.set_ylabel("Global Attention Weight Ratio", fontsize=12)

    if hist_ax is not None:
        # 使用二维直方图展示真实频数分布，不进行平滑。
        hist = sns.histplot(
            x=entropy,
            y=global_ratio,
            bins=(80, 80),
            stat="count",
            cmap="mako",
            cbar=True,
            ax=hist_ax,
        )
        if x_curve.size > 0:
            hist_ax.plot(x_curve, y_curve, color="#ffd43b", linewidth=2.0, label="Binned mean")
            hist_ax.legend(loc="best", frameon=True)
        hist_ax.set_title("2D Histogram", fontsize=14)
        hist_ax.set_xlabel("Token Prediction Entropy", fontsize=12)
        hist_ax.set_ylabel("Global Attention Weight Ratio", fontsize=12)

        # seaborn.histplot(cbar=True) 会创建 colorbar，这里补充标签便于论文图引用。
        if hist.collections:
            cbar = hist.collections[0].colorbar
            if cbar is not None:
                cbar.set_label("Count", fontsize=11)

    pearson = float(np.corrcoef(entropy, global_ratio)[0, 1]) if entropy.size > 1 else float("nan")
    fig.suptitle(f"{title} | N={entropy.size}, Pearson r={pearson:.4f}", fontsize=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot entropy/global-ratio relation (scatter / 2D histogram)")
    parser.add_argument("npy_path", type=str, help="Path to entropy_ratio_pairs.npy")
    parser.add_argument("--output", type=str, default="phase_transition_kde.png")
    parser.add_argument("--title", type=str, default="Entropy vs Global Attention Ratio")
    parser.add_argument("--mode", type=str, default="both", choices=["scatter", "hist", "both"])
    parser.add_argument("--only-updated", action="store_true", help="Use only rows with meta col2(is_updated)=1 when available")
    args = parser.parse_args()

    output = plot_entropy_kde(args.npy_path, args.output, args.title, args.mode, args.only_updated)
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
