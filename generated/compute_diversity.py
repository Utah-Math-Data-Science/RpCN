import pandas as pd
import torch
from PIL import Image
import clip
import argparse 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
model.eval()

k = 4
img_ids = pd.read_csv("prompt_ids.csv")["id"].tolist()

clip_diversities = []

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='v1.5', help='Model name')
parser.add_argument('--sampler', type=str, default='RpCN', help='Sampler name')
args = parser.parse_args()


with torch.no_grad():
    for img_id in img_ids:
        features = []

        for i in range(k):
            path = f"{args.model}/{args.sampler}/images/{img_id}_das_{i}.png"

            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=1, keepdim=True)
            features.append(feat)

        features = torch.cat(features, dim=0).float()

        dist_sq = torch.cdist(features, features) ** 2
        clip_div = dist_sq.triu(diagonal=1).sum() * (2 / (k * (k - 1)))

        clip_diversities.append(clip_div.item())

print(f'Model {args.model} average CLIP diversity: {sum(clip_diversities) / len(clip_diversities)}')
