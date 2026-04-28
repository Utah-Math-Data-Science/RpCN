from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from toy.train_toy.distributions import GaussianMixtureGenerator


DIR_ROOT = Path(__file__).resolve().parents[3]


def get_reward_gmm_and_subset(
    device,
    subset_indices=(0, 1, 4, 7),
    n_modes=8,
    radius=8,
    std=0.25,
    n_dims=2,
):
    gmm = GaussianMixtureGenerator(
        n_modes=n_modes,
        radius=radius,
        std=std,
        n_dims=n_dims,
        device=device,
    )
    subset_indices_tensor = torch.tensor(subset_indices, device=gmm.centers.device)
    return gmm, subset_indices_tensor


def plot_samples(samples, title="Samples from Gaussian Mixture on Hypersphere",
                 name='gaussian_mixture_samples.png'):

    fig, ax = plt.subplots(figsize=(10, 10))  # BIG figure

    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu())

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Ticks every 5
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))

    # Make tick labels MUCH larger
    ax.tick_params(axis='both', labelsize=20)

    # Title size
    ax.set_title(title, fontsize=24)

    ax.grid(True)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Remove extra whitespace
    plt.tight_layout()

    # Save high-resolution, tightly cropped
    plt.savefig(f'generated/{name}', dpi=300, bbox_inches='tight')
    plt.close()
