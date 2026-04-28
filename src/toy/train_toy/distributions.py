import torch


class GaussianGenerator:
    def __init__(self, mean=0.0, std=1.0, n_dims=2, device=None, seed=23):
        self.mean = mean
        self.std = std
        self.n_dims = n_dims
        self.device = device or torch.device("cpu")
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(seed)

    def generate(self, num_points):
        return (torch.randn(num_points, self.n_dims, device=self.device) * self.std + self.mean).float()



class GaussianMixtureGenerator:
    def __init__(self, n_modes=8, radius=5.0, std=0.5, n_dims=2, device=None, probs=None, seed=23):
        self.n_modes = n_modes
        self.radius = radius
        self.std = std
        self.n_dims = n_dims
        self.device = device or torch.device("cpu")
        self.probs = probs if probs is not None else torch.ones(n_modes, device=self.device) / n_modes
        self.probs = self.probs.to(self.device).float()
        self.probs = self.probs / self.probs.sum()
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(seed)


        angles = torch.linspace(0, 2 * torch.pi, n_modes + 1)[:-1]
        self.centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(self.device) * radius
        self.log_probs = torch.log(self.probs)
        self.log_norm_const = -0.5 * self.n_dims * torch.log(
            torch.tensor(2 * torch.pi * self.std**2, device=self.device)
        )

    def generate(self, num_points):
        mode_indices = torch.multinomial(self.probs, num_points, replacement=True).to(self.device)
        noise = torch.randn(num_points, self.n_dims, device=self.device) * self.std
        return (self.centers[mode_indices] + noise).float()

    def log_component_joint(self, x):
        x = x.to(self.device)
        x_exp = x.unsqueeze(1)
        centers_exp = self.centers.unsqueeze(0)
        sq_dist = ((x_exp - centers_exp) ** 2).sum(dim=-1)
        log_gauss = self.log_norm_const - 0.5 * sq_dist / (self.std**2)
        return log_gauss + self.log_probs.unsqueeze(0)

    def log_prob(self, x):
        return torch.logsumexp(self.log_component_joint(x), dim=1)
