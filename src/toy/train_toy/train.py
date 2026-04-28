import torch

from toy.train_toy.diffusion_trainer import DiffusionTrainer
from toy.train_toy.distributions import GaussianMixtureGenerator, GaussianGenerator
from toy.train_toy.model import DiffusionModel


def main() -> None:
    batch_size = 128
    num_batches_per_epoch = 64
    n_epochs = 1000
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim = 512
    n_hidden = 3


    n_steps = 200 # Diffusion trajectory steps
    betas = torch.linspace(1e-5, 0.01, n_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    p_init = GaussianGenerator()

    p_target = GaussianMixtureGenerator(8, 8, 0.25, 2)

    model = DiffusionModel(input_dim=2, output_dim=2, hidden_dim=512, n_hidden=3)

    trainer = DiffusionTrainer(model, p_init, p_target, batch_size, alphas_cumprod)

    # model_original = trainer_backward.train_mean_flow_model( num_batches_per_epoch, n_epochs, lr=lr)
    model = trainer.train_diffusion_model( num_batches_per_epoch, n_epochs, lr=lr)

    save_dict = {
            'model_state_dict': model.state_dict(),
            'alphas_cumprod': alphas_cumprod.cpu(),
            'betas': betas.cpu(),
            'config': {
                'dim': hidden_dim,
                'n_hidden': n_hidden,
                'time_embed_dim': 32
            }
        }

    torch.save(save_dict, "diffusion_gmm_model.pt")
    print("Model and schedule saved to diffusion_gmm_model.pt")


if __name__ == "__main__":
    main()
