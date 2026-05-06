import torch 
from rewards.DAPS_operators import get_operator
from utils import load_image_by_id
import numpy as np 
from torch.func import grad

def guidance_schedule(t, num_timesteps, start_frac=0.5):
    """
    Returns a scalar coeff in [0, 1] for guidance strength at timestep t.
    Reverse loop convention: t goes from (T-1) ... 0.
    Coeff increases as t decreases (later denoising => stronger guidance).
    """
    T = int(num_timesteps)
    if T <= 1:
        return 1.0
    # u: 0 at start (t=T-1), 1 at end (t=0)
    u = 1.0 - (float(t) / float(T - 1))
    u = max(0.0, min(1.0, u))

    # hard turn-on after halfway point 
    return 1.0 if u >= float(start_frac) else 0.0



class inverse_utils:
    def __init__(
        self,
        model,
        sampler,
        path_to_validation_imgs,
        batch_size = 1,
        resampling=10,
        guidance_schedule="late",
        grad_norm=True,
        image_id=60000,
        operator="gaussian_blur",
        device="cuda",
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device).float()
        self.sampler = sampler
        self.dtype =   torch.float32

        self.ref_image = load_image_by_id(path_to_validation_imgs, image_id, device=device, save=True).detach().to(self.device, self.dtype)

        self.operator = get_operator(operator)
        self.y = self.operator(self.ref_image).detach().to(self.device, self.dtype)

        self.batch_size = batch_size
        self.data_dim = (3, 256, 256)

        self.guidance_schedule = guidance_schedule
        self.grad_norm = grad_norm
        self.alphas_cumprod = torch.from_numpy(self.sampler.alphas_cumprod).to(self.device, self.dtype)        
        self.check_every = resampling
        self.timesteps = torch.arange(self.sampler.num_timesteps - 1, -1, -1)



    def make_t_batch(self, t, batch_size):
        if torch.is_tensor(t):
            return t.expand(batch_size).to(device=self.device, dtype=torch.long)
        return torch.full((batch_size,), int(t), device=self.device, dtype=torch.long)
    
    def reward_fn(self, latents): 
        return self.operator.log_likelihood(latents, self.y)
    
    def get_coeffecients_by_t(self, t, t_prev):
        
        alpha_bar_t = self.alphas_cumprod[t]
        # breakpoint()
        # If t_prev is negative (last step), alpha_bar_prev is 1.0
        alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
        
        # alpha_t is the ratio of the cumulative products
        alpha_t = alpha_bar_t / alpha_bar_t_prev
        beta_t = 1 - alpha_t
        
        return alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t

    @torch.enable_grad()
    def get_reward_gradient(self, xt, t):
        def scalar_reward(z):
            z = z
            t_batch = t.expand(1).long()

            x0 = self.get_tweedie(t_batch, z)
            return self.operator.log_likelihood(x0, self.y).sum()

        g = grad(scalar_reward)(xt)
        return g / g.flatten().norm().clamp_min(1e-8)
    

    def expected_next_fn(self, t, t_prev, x):
        x = x.to(device=self.device, dtype=self.dtype)

        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, t_prev)

        t_batch = self.make_t_batch(t, x.size(0))
        t_in = self.sampler._scale_timesteps(t_batch)

        noise_pred = self.model(x, t_in)

        noise_pred, predicted_variance = torch.chunk(noise_pred, 2, dim=1)

        mu = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )

        grad = self.get_reward_gradient(x, t_batch)

        return mu +  guidance_schedule(
            t, self.sampler.num_timesteps
        ) * grad


    def std_fn(self, t, t1=None):
        if t1 is None:
            return t * 0 + 1 
        else: 
            alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, t1)
            sigma_t = (((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t).sqrt()
            return sigma_t
            
    def get_tweedie(self,t, latents):
        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, 0)
        t_batch = self.make_t_batch(t, latents.size(0))


        noise_pred = self.model(latents, t_batch)
        noise_pred, predicted_variance = torch.chunk(noise_pred, 2, dim=1)

        return  (1/alpha_bar_t.sqrt()) * (latents - (1  - alpha_bar_t).sqrt() * noise_pred)



