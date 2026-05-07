import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np 


def forward_euler_simulator(model, source_samples, n_steps=50):

    device = next(model.parameters()).device
    xs = source_samples.to(device)
    B, d = xs.shape
    
    positions = torch.zeros(n_steps, B, d, device=device)
    positions[0] = xs

    t_space = torch.linspace(0.0, 1.0, n_steps, device=device)

    for i in range(1, n_steps):
        cur = positions[i-1]                       # <-- FIXED
        cur_t = t_space[i].unsqueeze(0)            # t = t_i
        cur_r = t_space[i-1].unsqueeze(0)          # r = t_{i-1}


        vel = model(cur,
                    cur_t.expand(B, 1),
                    cur_r.expand(B, 1))

        dt = t_space[i] - t_space[i-1]
        positions[i] = cur - dt * vel

    return t_space.cpu(), positions.cpu()

def plot_and_save_trajectories(
    model,
    n_samples,
    n_steps,
    save_dir,
    p_init,
    p_target,
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Sample from source distribution
    src = p_init.generate(n_samples)

    model_cpu = model.to("cpu")
    model_cpu.eval()

    # Run forward Euler transport
    t_space, positions = forward_euler_simulator(
        model_cpu, src, n_steps=n_steps
    )
    end_points = positions[-1].detach().numpy()

    # Sample from target distribution
    target_samples = p_target.generate(n_samples)

    # ---- Plotting ----
    plt.figure(figsize=(12, 5))

    # Plot 1: MeanFlow pushforward
    plt.subplot(1, 2, 1)
    plt.scatter(
        end_points[:, 0],
        end_points[:, 1],
        s=10,
        alpha=0.6,
    )
    plt.title("MeanFlow Pushforward (Forward Euler)")
    # plt.xlim([-1.5, 1.5])
    # plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect("equal")

    # Plot 2: Target distribution
    plt.subplot(1, 2, 2)
    plt.scatter(
        target_samples[:, 0],
        target_samples[:, 1],
        s=10,
        alpha=0.6,
        color="green",
    )
    plt.title("Target Distribution")
    # plt.xlim([-1.5, 1.5])
    # plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/meanflow_vs_target.pdf", dpi=200)
    plt.close()



def plot_target(p_target, n_samples=5000):
    plt.figure(figsize=(6, 6))
    x_samples = p_target.generate(n_samples).cpu().numpy()
    plt.scatter(x_samples[:, 0], x_samples[:, 1], s=2, alpha=0.3)
    plt.axis("equal")
    plt.title("Target Distribution")
    plt.savefig("target_distribution_circular_mixture.pdf", dpi=300)
