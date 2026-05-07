import torch
import torch.autograd.functional as func

def backward_loss(model, source_samples, target_samples, t, r):

    x_interp = (1 - t) * target_samples + t * source_samples
    velocity = source_samples - target_samples

    def f(x, t_val, r_val):
        return model(x, t_val, r_val)

    tangents = (
        velocity,                  # dx = velocity (full dim)
        torch.ones_like(t),        # dt = 1
        torch.zeros_like(r)        # dr = 0
    )

    u, dudt = func.jvp(f, (x_interp, t, r), tangents , create_graph=True)  # ← REQUIR)


    u_target = velocity - (t - r) * dudt

    loss =  torch.nn.functional.mse_loss(u, u_target.detach())


    return loss
