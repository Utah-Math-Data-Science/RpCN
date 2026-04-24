import torch
import torch.nn as nn

class LogReward(nn.Module):
    def __init__(self, centers, std, device="cuda"):
        """
        Args:
            centers: [K, D] tensor of means
            std: scalar (shared isotropic std)
            probs: [K] mixture weights (optional, defaults to uniform)
        """
        super().__init__()
        self.device = device

        self.std = std
        self.dim = 2
        self.centers = centers
        # Precompute normalization constant
        self.log_norm_const = -0.5 * self.dim * torch.log(
            torch.tensor(2 * torch.pi * std**2)
        ).to(self.device)

        self.log_probs = torch.log(torch.ones(centers.shape[0]) / centers.shape[0]).to(self.device)
    def forward(self, x):
        """
        Args:
            x: [B, D]
        Returns:
            log p(x): [B]
        """
        # [B, 1, D]
        x_exp = x.unsqueeze(1)

        # [1, K, D]
        centers_exp = self.centers.unsqueeze(0).to(self.device)

        # Squared distance: [B, K]
        sq_dist = ((x_exp - centers_exp) ** 2).sum(dim=-1)

        # Log Gaussian per component: [B, K]
        log_gauss = self.log_norm_const - 0.5 * sq_dist / (self.std**2)

        # Add mixture weights and log-sum-exp
        return torch.logsumexp(log_gauss + self.log_probs, dim=1)