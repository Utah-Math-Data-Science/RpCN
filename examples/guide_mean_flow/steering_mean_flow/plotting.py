from pathlib import Path
import matplotlib.pyplot as plt

def plot_steered_vs_unguided(x0, x_final, experiment):
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:, 0].cpu(), x0[:, 1].cpu(), s=2, alpha=0.3, label="Unguided")
    plt.scatter(x_final[:, 0], x_final[:, 1], s=2, alpha=0.3, label="Source Tempering")
    plt.legend()
    plt.axis("equal")
    plt.title("MeanFlow + Source Tempering (Orange) vs Unguided (Blue)")
    plt.savefig(Path(__file__).parent/f"results/imgs/{experiment}_steered_comparison.png", dpi=300)

