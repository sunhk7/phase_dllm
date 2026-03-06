import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_attention_dynamics(npy_path: str, output_path: str | None = None, title: str | None = None) -> str:
    dynamics = np.load(npy_path)
    if dynamics.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {npy_path}, got shape {dynamics.shape}")

    # Stored as (timesteps, layers); heatmap expects (layers, timesteps).
    heatmap_data = dynamics.T

    if output_path is None:
        output_path = str(Path(npy_path).with_suffix(".png"))
    if title is None:
        title = "LLaDA Spatio-Temporal Attention Dynamics Heatmap"

    fig_w = max(10, heatmap_data.shape[1] * 0.15)
    fig_h = max(4, heatmap_data.shape[0] * 0.3)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", cbar=True)
    ax.set_xlabel("Diffusion Timestep")
    ax.set_ylabel("Transformer Layers")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot LLaDA attention dynamics heatmap")
    parser.add_argument("npy_path", type=str, help="Path to .npy dynamics matrix")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    args = parser.parse_args()

    output_path = plot_attention_dynamics(args.npy_path, args.output, args.title)
    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
