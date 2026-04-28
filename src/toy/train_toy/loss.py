import torch
import torch.nn.functional as F

def diffusion_loss(model, x_0, t, alphas_cumprod):
    """
    Standard DDPM noise-prediction loss.
    
    Args:
        model: Your Diffusion Model (MLP)
        x_0: Clean data batch from target_dist [batch_size, 2]
        t: Random timesteps [batch_size, 1] (values 0 to 1)
        alphas_cumprod: Pre-calculated schedule tensor
    """
    # 1. Sample unit Gaussian noise
    noise = torch.randn_like(x_0)
    
    # 2. Map float t (0-1) to discrete indices for the alpha schedule
    num_steps = alphas_cumprod.size(0)
    t_idx = (t * (num_steps - 1)).long().view(-1)
    
    # 3. Grab schedule parameters
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t_idx]).view(-1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1. - alphas_cumprod[t_idx]).view(-1, 1)
    
    # x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*epsilon
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    # 5. Model predicts the noise we just added
    predicted_noise = model(x_t, t)

    # 6. Minimize MSE between added noise and predicted noise
    return F.mse_loss(predicted_noise, noise)