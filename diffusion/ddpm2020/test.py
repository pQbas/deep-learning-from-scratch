from parameters import alpha, alpha_hat, beta, T
import torch
import gc
from torchvision import transforms

import torch
from model import SimpleUnet, Block
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    model = SimpleUnet()
    model = torch.load('weights/weights.pt')
    model.eval()

    image2tensor = transforms.Compose([
            transforms.Resize(size = (224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])

    tensor2image = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
    ])

    imgsz = (1,3,640,640)
    sigma = torch.sqrt(beta).to('cuda')

    xt = torch.randn(imgsz).to('cuda')

    xt = torch.randn_like(xt, device='cuda')

    t_range = torch.arange(T).to('cuda')
    alpha_hat = alpha_hat.to('cuda')
    alpha = alpha.to('cuda')
    model = model.to('cuda')
    random = torch.randn_like(xt, device='cuda')

    xt = torch.randn_like(xt, device='cuda')
    plt.imshow(tensor2image(xt[0].cpu().detach()))
    plt.show()


    import torch
    t = torch.tensor([], dtype=torch.long, device=torch.device('cuda:0'))
    t_list = torch.arange(T, device='cuda')
    xt = torch.randn_like(xt, device='cuda')

    for t in reversed(t_list):

        t = t[None]
        with torch.no_grad():
            xt = 1/(torch.sqrt(alpha[t])) * (xt - model(xt, t)*(1 - alpha_hat[t])/torch.sqrt(1 - alpha_hat[t])) + sigma[t]*random

        plt.imshow(tensor2image(xt[0].cpu().detach()))
        plt.show()