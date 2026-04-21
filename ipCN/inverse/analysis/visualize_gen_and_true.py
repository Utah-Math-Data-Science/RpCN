from PIL import Image
import torch
from analysis_utils import  normalize_tensor, load_tensor_by_1_id, load_specific_image, load_folders_by_operator_and_id
import matplotlib.pyplot as plt

path_to_generated = '/mntb/jkellerData/inverse-ipcn/'
path_to_true = '~/'

if __name__ == '__main__':
    # Your loading code
    img_id= 60000
    database_id = load_folders_by_operator_and_id('phase_retrieval', img_id).values[0]
    generated_ndarray = normalize_tensor(load_tensor_by_1_id(path_to_generated, database_id))
    true_tensor = load_specific_image(img_id)

    # generated_img = generated_tensor.permute(1, 2, 0).cpu().detach().numpy()
    true_img = true_tensor.permute(1, 2, 0).cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(generated_ndarray)
    axes[0].set_title('Generated')
    axes[0].axis('off')

    axes[1].imshow(true_img)
    axes[1].set_title('True')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{img_id}_comparison.png')
