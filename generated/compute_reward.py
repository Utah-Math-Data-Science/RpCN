import os 
import pandas as pd
from importlib.resources import files
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='v1.5', help='Model name')
parser.add_argument('--sampler', type=str, default='RpCN', help='Sampler name')
args = parser.parse_args()

csv_dir = f"{args.model}/{args.sampler}/csv"

files = os.listdir(csv_dir)


newdf = pd.read_csv(files("rpcn.data").joinpath("prompt_ids.csv"))
total_ids = newdf['id'].tolist()




counter = 0 
score = []
gen_ids = []
for file in files: 
    df = pd.read_csv(os.path.join(csv_dir, file))
    image_id = file[:11]
    gen_ids.append(image_id)

    max_reward = df["reward"].max()
    score.append(max_reward)
    counter += 1



missing_ids = list(set(total_ids) - set(gen_ids))
print(f'missing ids: {missing_ids}')

print(f"Average reward for {args.model}: {sum(score)/len(score)}")   
print(f'number of files: {counter}')
