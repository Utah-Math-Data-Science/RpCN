import pandas as pd 
import torch
import os
import glob
from PIL import Image
from torchvision import transforms
import numpy as np 



# df = pd.read_csv('data.csv')

# def load_ids_by_operator(operator, num): 
#     return df.loc[df['operator'] == operator, 'alt_id'][:num]


# def get_multi_paired_df( operator, expected_count=2):
#     # 1. Filter by operator
#     subset = df[df['operator'] == operator].copy()
    
#     # 2. Rank each alt_id within its img_id group (0, 1, 2...)
#     # Sorting ensures that alt_id_1, _2, _3 are assigned consistently
#     subset = subset.sort_values(['img_id', 'alt_id'])
#     subset['occurrence'] = subset.groupby('img_id').cumcount() + 1
    
#     # 3. Pivot: Every occurrence becomes its own column
#     wide_df = subset.pivot(index='img_id', columns='occurrence', values='alt_id')
    
#     # 4. Clean up column names (e.g., 'alt_id_1', 'alt_id_2', 'alt_id_3')
#     wide_df.columns = [f'alt_id_{col}' for col in wide_df.columns]
#     wide_df = wide_df.reset_index()
    
#     # 5. Optional: Filter for only those that have all 3 IDs
#     # This removes any img_id that only has 1 or 2 entries
#     wide_df = wide_df.dropna(subset=[f'alt_id_{expected_count}'])
    
#     return wide_df

# def load_folders_by_operator_and_id(operator, id): 
#     return df.loc[(df['operator'] == operator) & (df['img_id'] == id), 'alt_id']



def load_tensors_by_id(path, blur_ids): 
    # Pre-allocate the 4D tensor
    tensors = torch.zeros((len(blur_ids), 3, 256, 256))
    
    # Use enumerate to get a clean index (0, 1, 2...) along with the ID
    for i, b_id in enumerate(blur_ids):
        folder_path = os.path.join(path, str(b_id))
        search_pattern = os.path.join(folder_path, "*.pt")
        files = glob.glob(search_pattern)
        
        if files:
            file_path = files[0]
            loaded_tensor = torch.load(file_path)
            
            # Place the loaded tensor into the i-th slot of the batch

            tensors[i] =  loaded_tensor
        else:
            print(f"Warning: No .pt file found for ID {b_id}")
            
    return tensors

def seed_function(seed, id):
    random = seed + 2435* id
    return random 

        

def load_true_images(id_range): 
    # Expand '~' to the full home directory path
    path = os.path.expanduser('~/.cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples')
    
    # Define the transformation: Convert PIL image to PyTorch Tensor
    # This also scales pixels to the range [0.0, 1.0]
    transform = transforms.ToTensor()
    
    image_tensors = []
    
    # Iterate through the range (0 to id_range - 1)
    for i in range(id_range):
        # Construct the filename. 
        # Note: Check if your files are named '0.png' or '00000.png' 
        # This assumes simple '0.png', '1.png', etc.
        img_name = f"{60000+ i}.png"
        img_path = os.path.join(path, img_name)
        
        if os.path.exists(img_path):
            # Open image and convert to RGB (to ensure 3 channels)
            with Image.open(img_path).convert('RGB') as img:
                tensor = transform(img)
                image_tensors.append(tensor)
        else:
            print(f"Warning: Image {img_path} not found.")
            
    # Stack list of tensors into a single 4D tensor: [N, C, H, W]
    if image_tensors:
        return torch.stack(image_tensors)

def normalize_tensor(x):
    img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img -= np.min(img)
    img /= np.max(img)
    return img




###########################################

def load_specific_image(id): 
    # Expand '~' to the full home directory path
    path = os.path.expanduser('~/.cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples')
    
    # Define the transformation: Convert PIL image to PyTorch Tensor
    # This also scales pixels to the range [0.0, 1.0]
    transform = transforms.ToTensor()
        
    # Iterate through the range (0 to id_range - 1)

    img_name = f"{id}.png"
    img_path = os.path.join(path, img_name)
        
    if os.path.exists(img_path):
        # Open image and convert to RGB (to ensure 3 channels)
        with Image.open(img_path).convert('RGB') as img:
            tensor = transform(img)
        return tensor.to('cpu')

    else:
        raise FileNotFoundError(f"Image {img_path} not found.")
            







def load_tensor_by_1_id(path, id): 

    folder_path = os.path.join(path, str(id))
    search_pattern = os.path.join(folder_path, "*.pt")
    files = glob.glob(search_pattern)
        
    if files:
        file_path = files[0]
        loaded_tensor = torch.load(file_path)
            
            # Place the loaded tensor into the i-th slot of the batch
    else:
        raise FileNotFoundError(f"No .pt file found for ID {id}")              
    return loaded_tensor.to('cpu')









@torch.no_grad()
def load_true_tensors_by_id(): 
    # Expand '~' to the full home directory path
    path = os.path.expanduser('~/.cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples')
    img_ids = pd.read_csv('../ids.csv')['id'].tolist()

    # Define the transformation: Convert PIL image to PyTorch Tensor
    # This also scales pixels to the range [0.0, 1.0]
    transform = transforms.ToTensor()
    
    image_tensors = []
    
    # Iterate through the range (0 to id_range - 1)
    for img_id in img_ids:
        # Construct the filename. 
        # Note: Check if your files are named '0.png' or '00000.png' 
        # This assumes simple '0.png', '1.png', etc.
        img_name = f"{img_id}.png"
        img_path = os.path.join(path, img_name)
        
        if os.path.exists(img_path):
            # Open image and convert to RGB (to ensure 3 channels)
            with Image.open(img_path).convert('RGB') as img:
                tensor = transform(img)
                image_tensors.append(tensor)
        else:
            print(f"Warning: Image {img_path} not found.")
            
    # Stack list of tensors into a single 4D tensor: [N, C, H, W]
    if image_tensors:
        return torch.stack(image_tensors)


@torch.no_grad()
def load_local_tensors_by_id(operator, seed, run=3): 
    path = "../generated/tensors/" 
    img_ids = pd.read_csv('../ids.csv')['id'].tolist()
    gen_tensors = []
    for img_id in img_ids:
        if run is not None: 
            file_name = f'{img_id}_{operator}_{seed_function(seed, img_id)}_{run}.pt'
        else: 
            file_name = f'{img_id}_{operator}_{seed_function(seed, img_id)}.pt'

        generated = torch.load(f'../generated/tensors/{file_name}')
        
  
        gen_tensors.append(generated)
    
    return torch.stack(gen_tensors)




@torch.no_grad()
def load_local_tensors(operator, seed): 
    path = "../generated/tensors/" 
    img_ids = pd.read_csv('../ids.csv')['id'].tolist()
    gen_tensors = []
    for img_id in img_ids:
        file_name = f'{img_id}_{operator}_{seed_function(seed, img_id)}.pt'

        generated = torch.load(f'../generated/tensors/{file_name}')
        
        gen_tensors.append(generated)
    
    return torch.stack(gen_tensors)