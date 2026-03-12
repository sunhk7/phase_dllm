#!/usr/bin/env python3
"""
Plot 3×3 Scatter Grid: Per-Token Entropy vs. Local Ratio (S_local)
===================================================================

Loads spatiotemporal attention weights saved as a dict of
{step_X_layer_Y: tensor(batch, heads, seq_len, seq_len)} from a .pt file,
computes per-token Entropy and Local Ratio for each (layer, step) combination,
and generates a 3×3 scatter plot grid.

Usage:
    python plot_entropy_vs_locality.py \
        --attn-path results/prompt/attn_weights_00000.pt \
        --prompt-length 128 \
        --window-size 64 \
        --output-path results/prompt/entropy_vs_locality_grid.png
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def compute_entropy_and_local_ratio(
    attn_weights: torch.Tensor,
    prompt_length: int,
    window_size: int,
):
    """
    For a single attention tensor, compute per-token Entropy and Local Ratio.

    Args:
        attn_weights: Tensor of shape (batch, heads, seq_len, seq_len)
        prompt_length: Number of prompt tokens (L_prompt)
        window_size: Local window size W

    Returns:
        entropies: 1D numpy array, one entropy value per target token
        local_ratios: 1D numpy array, one S_local value per target token
    """
    # Average across heads -> (batch, seq_len, seq_len)
    attn_avg = attn_weights.float().mean(dim=1)

    seq_len = attn_avg.shape[-1]
    num_targets = seq_len - prompt_length

    if num_targets <= 0:
        raise ValueError(
            f"No target tokens: seq_len={seq_len}, prompt_length={prompt_length}"
        )

    # Slice target queries only: i >= prompt_length
    target_attn = attn_avg[:, prompt_length:, :]  # (batch, num_targets, seq_len)

    # Zero out attention to prompt tokens (j < prompt_length)
    target_attn[:, :, :prompt_length] = 0.0

    # Re-normalize so remaining weights sum to 1.0
    row_sums = target_attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    P_target = target_attn / row_sums  # (batch, num_targets, seq_len)

    # --- Entropy: H = -sum(P * log(P + eps)) ---
    H = -(P_target * torch.log(P_target + 1e-9)).sum(dim=-1)  # (batch, num_targets)

    # --- Local Ratio: S_local = sum of P_target within [i - W/2, i + W/2] ---
    half_w = window_size // 2
    S_local = torch.zeros_like(H)

    for rel_idx in range(num_targets):
        abs_idx = prompt_length + rel_idx
        lo = max(abs_idx - half_w, prompt_length)
        hi = min(abs_idx + half_w + 1, seq_len)
        S_local[:, rel_idx] = P_target[:, rel_idx, lo:hi].sum(dim=-1)

    # Flatten across batch
    entropies = H.reshape(-1).numpy()
    local_ratios = S_local.reshape(-1).numpy()

    return entropies, local_ratios


def plot_3x3_scatter(
    data_dict: dict,
    output_path: str,
    window_size: int,
):
    """
    Create a 3×3 grid of scatter plots: Entropy (x) vs Local Ratio (y).

    Args:
        data_dict: {(step, layer): (entropies_array, local_ratios_array)}
        output_path: Path to save the PNG
        window_size: W value (for axis labels)
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey=True)

    unique_steps = sorted(set(k[0] for k in data_dict.keys()))
    unique_layers = sorted(set(k[1] for k in data_dict.keys()))

    # Color palette for visual distinction
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, layer in enumerate(unique_layers):
        for j, step in enumerate(unique_steps):
            ax = axes[i, j]
            key = (step, layer)
            if key not in data_dict:
                ax.set_visible(False)
                continue

            entropies, local_ratios = data_dict[key]

            # Scatter
            ax.scatter(
                entropies, local_ratios,
                alpha=0.35, s=8, color=colors[j % len(colors)],
                edgecolors="none", rasterized=True,
            )

            # Regression line
            try:
                z = np.polyfit(entropies, local_ratios, 1)
                p_line = np.poly1d(z)
                x_range = np.linspace(entropies.min(), entropies.max(), 100)
                ax.plot(x_range, p_line(x_range), color="#E53935", linewidth=2, alpha=0.8)
            except Exception:
                pass

            # Spearman correlation
            rho, p_val = spearmanr(entropies, local_ratios)

            ax.set_title(
                f"Layer {layer}, Step {step}\n"
                f"$\\rho={rho:.3f}$, n={len(entropies)}",
                fontsize=11, fontweight="bold",
            )
            ax.set_ylim(-0.05, 1.05)

            if i == len(unique_layers) - 1:
                ax.set_xlabel("Entropy $H$", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"Local Ratio $S_{{local}}$\n(W={window_size})", fontsize=11)

    plt.suptitle(
        "Per-Token Entropy vs. Local Ratio ($S_{local}$) — Spatiotemporal Grid",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved 3×3 scatter grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 3×3 scatter grid of per-token Entropy vs. Local Ratio"
    )
    parser.add_argument(
        "--attn-path", type=str, required=True,
        help="Path to saved attention weights .pt file (dict of step_X_layer_Y -> tensor)",
    )
    parser.add_argument(
        "--prompt-length", type=int, required=True,
        help="Number of prompt tokens (L_prompt)",
    )
    parser.add_argument(
        "--window-size", type=int, default=64,
        help="Local window size W for computing S_local (default: 64)",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Output PNG path (default: same dir as attn-path, named entropy_vs_locality_grid.png)",
    )
    args = parser.parse_args()

    # Default output path
    if args.output_path is None:
        import os
        base_dir = os.path.dirname(args.attn_path) or "."
        args.output_path = os.path.join(base_dir, "entropy_vs_locality_grid.png")

    print(f"[INFO] Loading attention weights from {args.attn_path} ...")
    weights_data = torch.load(args.attn_path, map_location="cpu", weights_only=False)

    if isinstance(weights_data, dict):
        print(f"[INFO] Detected spatiotemporal dict with {len(weights_data)} entries.")
        parsed_data = {}
        for key, tensor in weights_data.items():
            # key format: "step_X_layer_Y"
            parts = key.split("_")
            step = int(parts[1])
            layer = int(parts[3])

            print(f"  Processing {key} (shape={tensor.shape}) ...")
            ents, lrs = compute_entropy_and_local_ratio(
                tensor, args.prompt_length, args.window_size
            )
            parsed_data[(step, layer)] = (ents, lrs)
            print(f"    -> {len(ents)} tokens, H=[{ents.min():.3f}, {ents.max():.3f}], "
                  f"S_local=[{lrs.min():.3f}, {lrs.max():.3f}]")

        plot_3x3_scatter(parsed_data, args.output_path, args.window_size)

    elif isinstance(weights_data, torch.Tensor):
        print(f"[INFO] Single tensor detected (shape={weights_data.shape}).")
        print("[WARN] Single tensor only produces 1 subplot. Use spatiotemporal recording for 3×3 grid.")
        ents, lrs = compute_entropy_and_local_ratio(
            weights_data, args.prompt_length, args.window_size
        )
        # Plot as single scatter
        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ents, lrs, alpha=0.4, s=10, color="#4C72B0", edgecolors="none")
        rho, p_val = spearmanr(ents, lrs)
        ax.set_xlabel("Entropy $H$")
        ax.set_ylabel(f"Local Ratio $S_{{local}}$ (W={args.window_size})")
        ax.set_title(f"Entropy vs Local Ratio ($\\rho$={rho:.3f})")
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        fig.savefig(args.output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved single scatter plot to {args.output_path}")

    else:
        raise TypeError(
            f"Expected dict or torch.Tensor, got {type(weights_data)}. "
            "Make sure the .pt file was saved by spatiotemporal recording."
        )


if __name__ == "__main__":
    main()
