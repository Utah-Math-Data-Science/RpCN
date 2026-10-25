import torch 
from rewards.DAPS_operators import get_operator
from utils import load_image_by_id
import numpy as np 
from torch.func import grad


def guidance_schedule(t, num_timesteps, schedule):
    progress = 1.0 - float(t) / (num_timesteps - 1)
    if schedule == "none":
        return 0.0
    if schedule == "constant":
        return 1.0
    if schedule == "early":
        return 1.0 - progress
    if schedule == "middle":
        return 4.0 * progress * (1.0 - progress)
    if schedule == "late":
        return progress
    if schedule == "late_quadratic":
        return progress ** 2
    if schedule == "late_cubic":
        return progress ** 3
    if schedule == "late_quartic":
        return progress ** 4
    if schedule == "late_sextic":
        return progress ** 6
    if schedule == "late_octic":
        return progress ** 8
    if schedule == "last_half":
        return float(progress >= 0.5)
    raise ValueError(f"Unknown guidance schedule: {schedule}")





class inverse_utils:
    def __init__(
        self,
        model,
        sampler,
        path_to_validation_imgs,
        batch_size = 1,
        resampling=10,
        guidance_schedule="late",
        guidance_scale=0.03,
        guidance_scaling="std",
        grad_clip_norm=2.5,
        guidance_residual_floor=None,
        grad_norm=True,
        image_id=49000,
        operator="gaussian_blur",
        device="cuda",
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device).float()
        self.sampler = sampler
        self.dtype =   torch.float32

        self.ref_image = load_image_by_id(path_to_validation_imgs, image_id, device=device, save=True).detach().to(self.device, self.dtype)

        self.operator = get_operator(operator)
        # self.y = self.operator(self.ref_image).detach().to(self.device, self.dtype)
        self.y = self.operator.measure(self.ref_image).detach().to(self.device, self.dtype)
        self.batch_size = batch_size
        self.data_dim = (3, 256, 256)

        self.guidance_schedule = guidance_schedule
        self.guidance_scale = guidance_scale
        self.guidance_scaling = guidance_scaling
        self.grad_clip_norm = grad_clip_norm
        self.guidance_residual_floor = guidance_residual_floor
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
            t_batch = t.expand(1).long()
            x0 = self.get_tweedie(t_batch, z)
            return self.operator.log_likelihood(x0, self.y).sum()

        g = grad(scalar_reward)(xt)

        rms = g.flatten(1).square().mean(1).sqrt().clamp_min(1e-8)
        scale = (self.grad_clip_norm / rms).clamp(max=1.0)
        return g * scale.view(-1, *([1] * (g.ndim - 1)))

    def expected_next_fn(self, t, t_prev, x):
        x = x.to(device=self.device, dtype=self.dtype)

        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = (
            self.get_coeffecients_by_t(t, t_prev)
        )

        t_batch = self.make_t_batch(t, x.size(0))
        t_in = self.sampler._scale_timesteps(t_batch)
        noise_pred = self.model(x, t_in)
        noise_pred, _ = torch.chunk(noise_pred, 2, dim=1)
        mu = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )

        sigma = self.std_fn(t, t_prev)
        schedule_scale = guidance_schedule(
            t, self.sampler.num_timesteps, self.guidance_schedule
        )
        if schedule_scale == 0.0:
            return mu

        g = self.get_reward_gradient(x, t_batch)
        if self.guidance_residual_floor is not None:
            with torch.no_grad():
                x0 = self.get_tweedie(t_batch, x)
                residual = (self.operator(x0) - self.y).flatten(1).square().mean(1)
                guide = residual > self.guidance_residual_floor
            g = g * guide.view(-1, *([1] * (g.ndim - 1)))
        if self.guidance_scaling == "direct":
            noise_scale = 1.0
        elif self.guidance_scaling == "std":
            noise_scale = sigma
        else:
            noise_scale = sigma.square()
        return mu + self.guidance_scale * schedule_scale * noise_scale * g


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
        noise_pred, _ = torch.chunk(noise_pred, 2, dim=1)
        return (1 / alpha_bar_t.sqrt()) * (
            latents - (1 - alpha_bar_t).sqrt() * noise_pred
        )
