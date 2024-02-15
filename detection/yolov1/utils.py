import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import torch
import gc
import matplotlib.pyplot as plt
import numpy as np


transform = Compose([
    transforms.ToTensor(), # Scales data into [0,1]
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
])


def tensor2image(tensor):

    if tensor.dim() == 3:
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

    if tensor.dim() == 2:
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

    return reverse_transforms(tensor)

def plot_images(images_list, title):
    fig, axs = plt.subplots(1, len(images_list), figsize=(20, 5))
    for idx, xt in enumerate(images_list):
        axs[idx].set_title(r'$x_{t}$'.replace("t",f"{idx}"))
        axs[idx].imshow(tensor2image(xt))
    fig.tight_layout()
    plt.show()

def clean_gpu():
    for i in range(10):    
        torch.cuda.empty_cache()
        gc.collect()
