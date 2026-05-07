from pathlib import Path
from distributions import GaussianGenerator, CircleGenerator, GaussianMixtureGenerator
from model import MeanFlowModel
from trainer import Trainer
import torch
import torch.optim as optim
import tqdm as TQDM
import argparse

from inf_and_plot import plot_and_save_trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--distribution",
        help="please indicate target distribution",
        required=True,
        choices=["Mixture", "Circle"]
    )

    args =parser.parse_args()

    batch_size = 4096
    num_batches_per_epoch = 50
    n_epochs = 60
    p_init = GaussianGenerator(n_dims=2, noise_std=0.5)

    if args.distribution == "Mixture":
        p_target = GaussianMixtureGenerator(
            n_modes=4,
            radius=4.0,
            std=0.5
        )

    elif args.distribution == "Circle":  # Change from 'if' to 'elif'
        p_target = CircleGenerator(
            R=1.5,
            r=1.0,
            d=0.5
        )
    else:
        print(f'args.distribution: {args.distribution}')
        raise ValueError("Please indicate valid target distribution.")

    model = MeanFlowModel(input_dim=2, output_dim=2, dim=256, n_hidden=2)


    trainer_original = Trainer(model, p_init, p_target, batch_size)

    trainer_original.train_mean_flow_model(num_batches_per_epoch, n_epochs, lr=1e-3)

    torch.save(
        model.state_dict(),
        Path(__file__).parent/f'../steering_mean_flow/model/meanflow_{args.distribution}.pt'
    )





