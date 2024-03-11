
from torch.utils.data import Dataset, DataLoader
import torch
import os
from skimage import io, transform
import numpy
import matplotlib.pyplot as plt
import glob
import cv2
import random
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import copy
from skimage.util import random_noise

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


class Invert():
  def __call__(self, x):
    return torchvision.transforms.functional.invert(x)  # do not change the data type


class gaussianNoise():
    def __init__(self, p):
        self.probability = p

    def __call__(self, x):
        if torch.rand(1) > self.probability:
            return torch.tensor(random_noise(x.numpy(), mode='gaussian', mean=0, var=0.05, clip=True))
        return x


class saltPepperNoise():
    def __init__(self, p):
        self.probability = p

    def __call__(self, x):
        if torch.rand(1) > self.probability:
            return torch.tensor(random_noise(x.numpy(), mode='salt', amount=0.05))
        return x


class speackleNoise():
    def __init__(self, p):
        self.probability = p

    def __call__(self, x):
        
        if torch.rand(1) > self.probability:
            return torch.tensor(random_noise(x.numpy(), mode='speckle', mean=0, var=0.05, clip=True))
        return x



transformation_defaul = T.Compose([
                        T.Resize(size = (224,224)),
                        T.ToTensor(),
                        ])

transformation_positive = T.Compose([
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomRotation(degrees=45),
                        speackleNoise(p=0.5),
                        saltPepperNoise(p=0.5),
                        gaussianNoise(p=0.5)
                        ])

transformation_negative = T.Compose([
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomRotation(degrees=45)
                        ])


class BLUEBERRYDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Read the root_dir path and transform
        self.root_dir = root_dir
        self.transform = transform
        self.samples_per_clase = 10
        self.num_clases = 6

        # Read the files in the dataset folder, and group them into lists of 24 elements
        data_path = os.path.join(self.root_dir,'*g')
        self.images = sorted(glob.glob(data_path))
        print(self.images)
        self.images_ordered = [self.images[i:i + 24] for i in range(0, len(self.images), 24)]

        print(self.images_ordered)

        # This transform is used in experiments
        self.transform = transformation_defaul

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, indice_class_sample):

        clase, sample = indice_class_sample
        if clase > self.num_clases or sample > self.samples_per_clase:
            return None # Use a warning instead of a NoneType
        
        # Get image name to open it with PIL
        img_name = os.path.join(self.images_ordered[clase][sample])
        
        # Reading the image using PIL
        image = Image.open(img_name)
        sample = {'image': image, 'category': clase}
        
        # If there's a transform then apply it over the image
        if self.transform:           
            sample['image'] = self.transform(sample['image'])

        return sample



class tripletMatcher(Dataset):
    def __init__(self, root, indices_list, transform=None):
        super(tripletMatcher, self).__init__()
        self.dataset = CEDARDataset(root, transform=transform)
        self.num_classes = self.dataset.num_clases
        self.num_samples = self.dataset.samples_per_clase
        self.indices_list = [x % self.num_classes for x in indices_list]

    def get_subset(self, new_indices_list):
        dataset_copy = copy.copy(self)
        dataset_copy.setNewIndicesList(new_indices_list)
        return
    
    def setNewIndicesList(self, new_indices_list):
        self.indices_list = [x % self.num_classes for x in new_indices_list]
        return
    
    def get_random_class(self):
        idx = random.randint(0, len(self.indices_list) - 1)
        return self.indices_list[idx]
    
    def get_random_sample(self):
        return random.randint(0, self.num_samples - 1)
    
    def __len__(self):
        return len(self.indices_list)
    
    def __getitem__(self, index):

        anchor = {
            "class_idx": self.get_random_class(),
            "sample_idx": self.get_random_sample()
        }
        anchor["image"] = self.dataset[(anchor["class_idx"], anchor["sample_idx"])]["image"]

       
        positive = {
            "class_idx": anchor["class_idx"],
            "sample_idx": self.get_random_sample()
        }
        while positive["sample_idx"] == anchor["sample_idx"]:
            positive["sample_idx"] = self.get_random_sample()
        positive["image"] = self.dataset[(positive["class_idx"], positive["sample_idx"])]["image"]


        negative = {
            "class_idx": self.get_random_class(),
            "sample_idx": self.get_random_sample()
        }
        while negative["class_idx"] == anchor["class_idx"]:
            negative["class_idx"] = self.get_random_class()
        negative["image"] = self.dataset[(negative["class_idx"], negative["sample_idx"])]["image"]


        ## Performing Augmentation:
        positive_ = transformation_positive(positive["image"])
        negative_ = transformation_negative(negative["image"])


        sample = {
            "anchor": anchor["image"],
            "positive": positive_,
            "negative": negative_
        }

        return sample


class datasetHandler():
    def __init__(self, root, dataset_size=500, batch_size=16):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.indices_list = range(0, dataset_size)
        self.root = root

    def get_train_test_set(self, dataloader, test_percentage):
        
        train_idx, test_idx, _, _ = train_test_split(
            self.indices_list,
            self.indices_list,
            test_size=test_percentage,
            random_state=42
        )

        trainLoader = DataLoader(
            tripletMatcher(root=self.root, indices_list=train_idx), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        testLoader = DataLoader(
            tripletMatcher(root=self.root, indices_list=test_idx), 
            batch_size=self.batch_size, 
            shuffle=True
        )

        return trainLoader, testLoader




if __name__ == '__main__':

    #data = datasetHandler()
    #trainloder, testloader = data.get_train_test_set(dataloader=None, test_percentage=0.3)

    #for i in trainloder:
    #    print(i["anchor"].shape)

    data = BLUEBERRYDataset(root_dir='./blueberry', transform=transformation_defaul)
    for i in data:
        print(i)