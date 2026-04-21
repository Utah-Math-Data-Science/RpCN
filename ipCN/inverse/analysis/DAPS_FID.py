import torch 
import torch.nn.functional as F
from piq import FID
from torchvision.models import inception_v3
import os
import glob
import pandas as pd 

from analysis_utils import get_multi_paired_df, load_ids_by_operator, load_tensors_by_id, load_true_images, normalize_tensor




def preprocess(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * 0.5 + 0.5
    images = (images - mean) / std
    return images.float()


def calculate_fid(real_loader, generated_loader, device='cuda'):
    # Load pretrained Inception model
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Modify the model to output features from the pre-logits layer
    model.to(device)
    model.eval()

    # Function to get features from a dataloader
    def get_features(loader):
        features = []
        for images in loader:
            images = preprocess(images)
            with torch.no_grad():
                images = images.to(device)
                output = model(images)
                features.append(output.cpu())
        features = torch.cat(features, dim=0)
        return features

    # Extract features
    real_features = get_features(real_loader)
    gen_features = get_features(generated_loader)

    # Compute FID score
    fid = FID()
    score = fid.compute_metric(real_features, gen_features)
    return score



path_to_generated = '/mntb/jkellerData/inverse-ipcn/'
path_to_true = '~/'


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++
    new_df = get_multi_paired_df('high_dynamic_range')
    ids_1 = new_df['alt_id_1'][:100].tolist()
    ids_2 = new_df['alt_id_2'][:100].tolist()
    ids_3 = new_df['alt_id_3'][:100].tolist()

    blur_ids = load_ids_by_operator('inpainting_random', 100)
    generated_tensors = load_tensors_by_id(path_to_generated, ids_3)
    true_tensors = load_true_images(100)


    fid_score = calculate_fid([generated_tensors], [true_tensors])
    print(f'FID Score: {fid_score.item():.4f}')



#47.7352, 44.2459, 43.6200