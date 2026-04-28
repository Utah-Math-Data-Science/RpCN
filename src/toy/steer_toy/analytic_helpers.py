import torch


class analytic_helpers():
    def __init__(self, alphas_cumprod, model, batch_size=256, device="cuda"):
        self.device = device
        self.model = model 
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.batch_size = batch_size
        self.n_steps = len(self.alphas_cumprod)
        self.check_every = 5
        self.data_dim = 2
        self.dtype = torch.float32
        self.timesteps =   torch.arange(len(self.alphas_cumprod) - 1, -1, -1)
        


    def get_coeffecients_by_t(self, t, t_prev):
        
        alpha_bar_t = self.alphas_cumprod[t]
        # breakpoint()
        # If t_prev is negative (last step), alpha_bar_prev is 1.0
        alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
        
        # alpha_t is the ratio of the cumulative products
        alpha_t = alpha_bar_t / alpha_bar_t_prev
        beta_t = 1 - alpha_t
        
        return alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t
    
    def expected_next_fn(self, t, t_prev, latents):
        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, t_prev)

        t_cont = torch.full(
                (self.batch_size, 1),
                t / (self.n_steps - 1),
                device=self.device,
                dtype=self.dtype,
        )

        pred_noise = self.model(latents, t_cont)

        return  (1.0 / torch.sqrt(alpha_t)) * (
            latents - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

    @torch.no_grad()
    def get_tweedie(self,t, latents):
        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, 0)
        t_cont = torch.full(
                (self.batch_size, 1),
                t / (self.n_steps - 1),
                device=self.device,
                dtype=self.dtype,
        )

        noise_pred = self.model(latents, t_cont)
        return  (1/alpha_bar_t.sqrt()) * (latents - (1  - alpha_bar_t).sqrt() * noise_pred)



    def std_fn(self, t, t1=None):
        if t1 is None:
            return t * 0 + 1 
        else: 
            alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, t1)
            sigma_t = (((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t).sqrt()
            return sigma_t
        

    @torch.no_grad()
    def particle_sampling(self):
        device = next(self.model.parameters()).device


        x = torch.randn(self.batch_size, 2, device=device)

        n_steps = len(self.alphas_cumprod)

        for t_idx in self.timesteps:

            mean = self.expected_next_fn(t_idx, t_idx-1, x)
            std=     self.std_fn(t_idx, t_idx - 1)
            tweedie = self.get_tweedie(t_idx, x)

            if t_idx > 0:
                noise = torch.randn_like(x)


                x = mean + std * noise
            else:
                x = mean

        return x.view(self.batch_size, 2)
