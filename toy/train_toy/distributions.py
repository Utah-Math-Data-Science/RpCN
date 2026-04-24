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
        self.probs = probs if probs is not None else torch.ones(n_modes) / n_modes
        self.seed = seed 
        if self.seed is not None:
            torch.manual_seed(seed) 


        angles = torch.linspace(0, 2 * torch.pi, n_modes + 1)[:-1]
        self.centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(self.device) * radius

    def generate(self, num_points):
        mode_indices = torch.multinomial(self.probs, num_points, replacement=True).to(self.device)
        noise = torch.randn(num_points, self.n_dims, device=self.device) * self.std
        return (self.centers[mode_indices] + noise).float()