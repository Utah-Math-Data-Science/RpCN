
from ipCN_DAPS_integrated import NoisePCNSampler
from utils import load_model_and_sampler
import numpy as np 
import yaml
import torch
import argparse 
from generated.samples.view_samples import plot_samples
import datetime 
import conf 

from utils import normalize_tensor






#seed for nonlinear 31, 63, 124


parser = argparse.ArgumentParser()
parser.add_argument('--rng_seed', type=int, default=124, help="Set a random seed")
parser.add_argument('--img_id', type=int, default=60075, help="Which validation image do you want to sample?")
parser.add_argument('--type', type=str, default = 'Linear', help="Are you solving a linear or nonlinear problem?")
parser.add_argument('--operator', type=str, default="phase_retrieval", help="What forward operator are you using?")
args = parser.parse_args()


def seed_function(seed, id):
    random = seed + 2435* id
    return random 





with conf.Session() as db:
 
    randomized_seed = seed_function(args.rng_seed, args.img_id)
    torch.manual_seed(randomized_seed)

    cfg = dict(
        _target_='conf.Conf',
        rng_seed=randomized_seed,
        img_id=args.img_id,
        operator=args.operator,
    )
    
    cfg: conf.Conf = conf.orm.instantiate_and_insert_config(db, cfg)

    db.commit()



    model, ddpm = load_model_and_sampler()

# for nonnormalized nonlinear tasks

    # all tasks except box inpainting and phase retrieval use this 
    # ipCN_sampler = NoisePCNSampler(model,ddpm, 4, 1, 25, 1.0, "late", image_id= args.img_id, operator=args.operator)


    ipCN_sampler = NoisePCNSampler(model,ddpm, 4, 1, 25, 1.0, "linear", image_id= args.img_id, operator=args.operator)

    start = datetime.datetime.now()
    x, rewards = ipCN_sampler.transport_parallel()




    # torch.save(x[-1],  cfg.run_dir/f'{args.img_id}_{args.operator}.pt')




    img = normalize_tensor(x[-1])

    plot_samples(img, args.img_id, args.operator)