#!/usr/bin/env python3
"""
Offline Analysis Script: Target-Centric Entropy vs. Locality
=============================================================

Loads saved attention weights from a .pt file of shape (batch, heads, seq_len, seq_len),
then performs correlation analysis between Information Entropy and Local Attention Mass
for Target Tokens only.

Usage:
    python analyze_attention.py \
        --attn-path saved_attn_weights.pt \
        --prompt-length 128 \
        --window-size 64 \
        --output-path entropy_vs_locality.png
"""

import argparse
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def analyze_target_attention(
    attn_weights: torch.Tensor,
    prompt_length: int,
    window_size: int,
):
    """
    Analyze target-to-target attention patterns.

    Args:
        attn_weights: Tensor of shape (batch, heads, seq_len, seq_len)
        prompt_length: Number of prompt tokens
        window_size: Local window size W

    Returns:
        entropies: 1D numpy array of entropy values for each target token
        local_masses: 1D numpy array of local attention mass for each target token
    """
    # Average across heads -> (batch, seq_len, seq_len)
    attn_avg = attn_weights.float().mean(dim=1)

    seq_len = attn_avg.shape[-1]
    num_target_tokens = seq_len - prompt_length

    if num_target_tokens <= 0:
        raise ValueError(
            f"No target tokens: seq_len={seq_len}, prompt_length={prompt_length}"
        )

    # Filter: only query tokens where i >= prompt_length
    # Shape: (batch, num_target, seq_len)
    target_attn = attn_avg[:, prompt_length:, :]

    # Re-normalization for Target-to-Target Attention:
    # Zero out weights where j < prompt_length (ignore prompt attention)
    target_attn[:, :, :prompt_length] = 0.0

    # Re-normalize so remaining weights sum to 1.0
    row_sums = target_attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    P_target = target_attn / row_sums  # (batch, num_target, seq_len)

    # Calculate Information Entropy H = -sum(P * log(P + 1e-9))
    H = -(P_target * torch.log(P_target + 1e-9)).sum(dim=-1)  # (batch, num_target)

    # Calculate Target Local Attention Mass S_local
    # For each target query at index i (absolute), sum P_target within [i - W/2, i + W/2]
    half_w = window_size // 2
    S_local = torch.zeros_like(H)

    for rel_idx in range(num_target_tokens):
        abs_idx = prompt_length + rel_idx
        lo = max(abs_idx - half_w, prompt_length)
        hi = min(abs_idx + half_w + 1, seq_len)
        S_local[:, rel_idx] = P_target[:, rel_idx, lo:hi].sum(dim=-1)

    # Flatten across batch
    entropies = H.reshape(-1).numpy()
    local_masses = S_local.reshape(-1).numpy()

    return entropies, local_masses


def plot_scatter(
    entropies: np.ndarray,
    local_masses: np.ndarray,
    output_path: str,
    window_size: int,
):
    """Create a scatter plot of Entropy vs. Local Attention Mass with regression."""
    # Compute Spearman correlation
    rho, p_value = spearmanr(entropies, local_masses)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(
        x=entropies,
        y=local_masses,
        scatter_kws={"alpha": 0.4, "s": 10, "color": "#4C72B0"},
        line_kws={"color": "#C44E52", "linewidth": 2},
        ax=ax,
    )

    ax.set_xlabel("Target-only Information Entropy (H)", fontsize=13)
    ax.set_ylabel(f"Target Local Attention Mass (S_local, W={window_size})", fontsize=13)
    ax.set_title("Target-Centric Entropy vs. Locality", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    # Display Spearman correlation
    textstr = f"Spearman ρ = {rho:.4f}\np-value = {p_value:.2e}"
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved scatter plot to {output_path}")
    print(f"[INFO] Spearman rho={rho:.4f}, p-value={p_value:.2e}")

def plot_scatter_grid(
    data_dict: dict,
    output_path: str,
    window_size: int,
):
    """Create a 3x3 grid of scatter plots for (layer, step) combinations."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

    # Sort layers and steps
    unique_steps = sorted(list(set([k[0] for k in data_dict.keys()])))
    unique_layers = sorted(list(set([k[1] for k in data_dict.keys()])))

    for i, layer in enumerate(unique_layers):
        for j, step in enumerate(unique_steps):
            ax = axes[i, j]
            if (step, layer) not in data_dict:
                ax.set_visible(False)
                continue

            entropies, local_masses = data_dict[(step, layer)]
            rho, p_value = spearmanr(entropies, local_masses)

            sns.regplot(
                x=entropies,
                y=local_masses,
                scatter_kws={"alpha": 0.3, "s": 5, "color": "#4C72B0"},
                line_kws={"color": "#C44E52", "linewidth": 2},
                ax=ax,
            )

            ax.set_title(f"Layer {layer}, Step {step}\n$\\rho={rho:.3f}$", fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            
            if i == 2:
                ax.set_xlabel("Target-only Entropy (H)")
            if j == 0:
                ax.set_ylabel(f"Local Mass (W={window_size})")

    plt.suptitle("Target-Centric Entropy vs. Locality across Layers and Steps", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved 3x3 scatter grid to {output_path}")

def plot_mass_distribution_grid(
    data_dict: dict,
    output_path: str,
    window_size: int,
):
    """Create a 3x3 grid of density plots (KDE/histogram) for local masses."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

    # Sort layers and steps
    unique_steps = sorted(list(set([k[0] for k in data_dict.keys()])))
    unique_layers = sorted(list(set([k[1] for k in data_dict.keys()])))

    for i, layer in enumerate(unique_layers):
        for j, step in enumerate(unique_steps):
            ax = axes[i, j]
            if (step, layer) not in data_dict:
                ax.set_visible(False)
                continue

            _, local_masses = data_dict[(step, layer)]

            sns.histplot(
                local_masses, 
                bins=30, 
                kde=True, 
                color="#2CA02C", 
                stat="density",
                alpha=0.4,
                ax=ax,
                edgecolor="None"
            )
            
            mean_val = local_masses.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}")
            ax.legend(loc='upper left', fontsize=9)

            ax.set_title(f"Layer {layer}, Step {step}", fontsize=12)
            ax.set_xlim(-0.05, 1.05)
            
            if i == 2:
                ax.set_xlabel(f"Local Mass (W={window_size})")
            if j == 0:
                ax.set_ylabel("Density")

    plt.suptitle("Distribution of Target Local Attention Mass across Layers and Steps", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved 3x3 distribution grid to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze saved attention weights: Target-Centric Entropy vs. Locality"
    )
    parser.add_argument(
        "--attn-path",
        type=str,
        required=True,
        help="Path to saved attention weights .pt file (shape: batch, heads, seq_len, seq_len)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        required=True,
        help="Number of prompt tokens",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Local window size W for computing S_local (default: 64)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="entropy_vs_locality.png",
        help="Output path for the scatter plot (default: entropy_vs_locality.png)",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading attention weights from {args.attn_path} ...")
    weights_data = torch.load(args.attn_path, map_location="cpu", weights_only=True)

    if isinstance(weights_data, torch.Tensor):
        # Single tensor case
        print(f"[INFO] Attention weights shape: {weights_data.shape}")
        print(f"[INFO] prompt_length={args.prompt_length}, window_size={args.window_size}")

        entropies, local_masses = analyze_target_attention(
            weights_data, args.prompt_length, args.window_size
        )

        print(
            f"[INFO] Analyzed {len(entropies)} target token data points "
            f"(entropy range: [{entropies.min():.4f}, {entropies.max():.4f}])"
        )
        plot_scatter(entropies, local_masses, args.output_path, args.window_size)
    
    elif isinstance(weights_data, dict):
        # Multi-layer/step case
        print("[INFO] Detected dict of attention weights (Spatiotemporal Analysis).")
        parsed_data = {}
        for key, tensor in weights_data.items():
            # key looks like "step_X_layer_Y"
            parts = key.split("_")
            step = int(parts[1])
            layer = int(parts[3])
            
            ents, masses = analyze_target_attention(tensor, args.prompt_length, args.window_size)
            parsed_data[(step, layer)] = (ents, masses)
            
        print(f"[INFO] Computed entropies and masses for {len(parsed_data)} combinations.")
        # 1. Plot scatter grid
        plot_scatter_grid(parsed_data, args.output_path, args.window_size)
        
        # 2. Plot distribution grid
        dist_output = args.output_path.replace("entropy_vs_locality", "local_mass_distribution")
        if dist_output == args.output_path:
            dist_output = args.output_path.replace(".png", "_dist.png")
        plot_mass_distribution_grid(parsed_data, dist_output, args.window_size)
    
    else:
        raise TypeError(
            f"Expected a torch.Tensor or dict, got {type(weights_data)}. "
            "Make sure you saved attention weights with torch.save()."
        )


if __name__ == "__main__":
    main()
