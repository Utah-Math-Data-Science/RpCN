import torch
import numpy as np

class GaussianGenerator:
    def __init__(self, n_dims=2, noise_std=1.0):
        self.n_dims = n_dims
        self.noise_std = noise_std

    def generate(self, num_points):
        return torch.randn(num_points, self.n_dims) * self.noise_std

class CircleGenerator:
    def __init__(self, R=1.0, r=0.6, d=0.5):
        self.R = R  # Outer radius
        self.r = r  # Inner circle radius
        self.d = d  # Offset of inner circle

    def generate(self, num_points):
        # Calculate the area ratio to estimate required samples
        outer_area = np.pi * self.R**2
        inner_area = np.pi * self.r**2
        crescent_area = outer_area - inner_area

        # Estimate required samples with 20% buffer
        n_samples = int(num_points * (outer_area / crescent_area) * 1.2)
        n_samples = max(n_samples, num_points)  # Ensure we generate at least num_points

        # Generate points in the outer circle
        theta = 2 * np.pi * torch.rand(n_samples)
        radius = self.R * torch.sqrt(torch.rand(n_samples))

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        # Filter points that are outside the inner circle
        mask = (x - self.d)**2 + y**2 > self.r**2
        points = torch.stack((x[mask], y[mask]), dim=1)

        # If we didn't get enough points, recursively generate more
        while points.shape[0] < num_points:
            additional_points = self.generate(num_points - points.shape[0])
            points = torch.cat((points, additional_points), dim=0)

        return points[:num_points].to(dtype=torch.float32)


class GaussianMixtureGenerator:
    """
    Generates points from a Mixture of Gaussians arranged in a circle.
    Useful for testing multimodal exploration in Source Tempering.
    """
    def __init__(self, n_modes=8, radius=5.0, std=0.5):
        """
        Args:
            n_modes (int): Number of Gaussian blobs (n).
            radius (float): Distance from the origin to each blob center.
            std (float): Standard deviation of each Gaussian blob.
        """
        self.n_modes = n_modes
        self.radius = radius
        self.std = std
        self.n_dims = 2

    def generate(self, num_points):
        # 1. Determine which mode each point belongs to
        # We sample mode indices uniformly from 0 to n_modes - 1
        mode_indices = torch.randint(0, self.n_modes, (num_points,))

        # 2. Calculate the centers for each mode
        # Angles are evenly spaced: 0, 2pi/n, 4pi/n, ...
        angles = torch.linspace(0, 2 * np.pi, self.n_modes + 1)[:-1]
        centers_x = self.radius * torch.cos(angles)
        centers_y = self.radius * torch.sin(angles)
        centers = torch.stack((centers_x, centers_y), dim=1)

        # 3. Assign each point to its corresponding center
        point_centers = centers[mode_indices]

        # 4. Add Gaussian noise around the centers
        noise = torch.randn(num_points, self.n_dims) * self.std
        points = point_centers + noise

        return points.to(dtype=torch.float32)

