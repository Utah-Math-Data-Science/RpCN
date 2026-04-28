import torch
import torch.nn as nn

from toy.train_toy.distributions import GaussianMixtureGenerator


class LogReward(nn.Module):
    r"""Log reward for subset posterior mass under a Gaussian mixture.

    Given a Gaussian mixture with latent component index ``j`` and an index subset
    ``S``, this module returns :math:`\log p(j \in S \mid x)`.

    The computation is

    .. math::

       \log p(j \in S \mid x)
       = \log \sum_{j \in S} p(j, x) - \log p(x).

    Example:
        >>> import torch
        >>> from toy.train_toy.distributions import GaussianMixtureGenerator
        >>> gmm = GaussianMixtureGenerator(n_modes=8, radius=8, std=0.25, device="cpu")
        >>> reward = LogReward(gmm, [0, 1, 4, 7])
        >>> x = torch.randn(16, 2)
        >>> y = reward(x)
        >>> y.shape
        torch.Size([16])
    """

    def __init__(self, gmm: GaussianMixtureGenerator, subset_indices):
        super().__init__()
        self.gmm = gmm
        self.device = gmm.device

        subset_indices_tensor = torch.as_tensor(subset_indices, device=self.device)
        if subset_indices_tensor.ndim != 1:
            raise ValueError("subset_indices must be a 1D sequence")
        if subset_indices_tensor.numel() == 0:
            raise ValueError("subset_indices cannot be empty")

        if torch.is_floating_point(subset_indices_tensor):
            rounded = subset_indices_tensor.round()
            if not torch.allclose(subset_indices_tensor, rounded):
                raise ValueError("subset_indices must contain integers")

        subset_indices_tensor = subset_indices_tensor.long()
        if (subset_indices_tensor < 0).any() or (subset_indices_tensor >= self.gmm.n_modes).any():
            raise ValueError("subset_indices must be in [0, n_modes)")

        self.subset_indices = subset_indices_tensor

    def forward(self, x):
        r"""Return :math:`\log p(j \in S \mid x)` for each sample in ``x``.

        Args:
            x: Tensor of shape ``[B, D]``.

        Returns:
            Tensor of shape ``[B]`` with values in ``(-inf, 0]``.
        """
        log_joint = self.gmm.log_component_joint(x)
        log_numerator = torch.logsumexp(log_joint[:, self.subset_indices], dim=1)
        log_denominator = self.gmm.log_prob(x)
        return log_numerator - log_denominator
