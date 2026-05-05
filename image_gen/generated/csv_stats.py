import os 
import pandas as pd 

model = "v1.5"
csv_dir = f"{model}/csv"

files = os.listdir(csv_dir)


newdf = pd.read_csv('.../prompts/prompt_ids.csv')
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

print(f"Average reward for {model}: {sum(score)/len(score)}")   
print(f'number of files: {counter}')