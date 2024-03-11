import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms

scheduler = lambda timesteps: torch.linspace(start=0.0001, end=0.02, steps=timesteps)

T = 100
beta = scheduler(T)

alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, axis=0)