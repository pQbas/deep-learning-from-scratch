import torch
from torch.optim import Adam
from model import SimpleUnet
from tqdm import tqdm
import gc
from dataloader import dataset
import torch.nn.functional as F
from parameters import alpha, alpha_hat, T


if __name__ == '__main__':

  for i in range(10):    
      torch.cuda.empty_cache()
      gc.collect()

  model = SimpleUnet()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  optimizer = Adam(model.parameters(), lr=0.001)
  num_epochs = 50

  for epoch in range(num_epochs):
    running_loss = 0.
    last_loss = 0.

    for i, batch in enumerate(tqdm(dataset)):

      # Zero your gradients for every batch!
      optimizer.zero_grad()
      
      # sample images from dataset
      clean_images, classes = batch[0].to(device), batch[1].to(device)

      # uniforme sample of timesteps
      bs = clean_images.shape[0]
      timesteps = torch.randint(0, T, (bs,), device=clean_images.device, dtype=torch.int64)

      # sample noise
      noise = torch.randn_like(clean_images, device=device) #torch.randn(clean_images.shape, device=device)

      # compute the noise images
      alpha_hat = alpha_hat.to(device)
      alpha_hat_ = alpha_hat[timesteps][...,None,None,None]
      noisy_images = torch.sqrt(alpha_hat_)*clean_images + torch.sqrt(1 - alpha_hat_)*noise
      
      # compute the noise predictions
      noise_pred = model(noisy_images, timesteps)

      # compare the noise from both and reduce
      loss = F.mse_loss(noise_pred, noise)
      loss.backward()

      # Adjust learning weights
      optimizer.step()

      # Gather data and report
      running_loss += loss.item()

    print(f"it:{epoch}/{num_epochs}, Average Loss:{running_loss/(i+1):.3f}")

  torch.save(model,'weights/weights.pt')