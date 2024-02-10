import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

transform = Compose([
    transforms.ToTensor(), # Scales data into [0,1]
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
])