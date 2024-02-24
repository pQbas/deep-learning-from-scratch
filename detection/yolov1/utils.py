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
        
        if type(xt) == torch.Tensor:
            axs[idx].imshow(tensor2image(xt))
    fig.tight_layout()
    plt.show()

def clean_gpu():
    for i in range(10):    
        torch.cuda.empty_cache()
        gc.collect()


import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
import gc

def get_dataset(dataset_name, transform, batchsize):
    
    if dataset_name == 'flowers102':

        data_transform = transforms.Compose(transform)

        train = torchvision.datasets.Flowers102(root="./datasets/", download=True,
                                        transform=data_transform)

        trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        test = torchvision.datasets.Flowers102(root="./datasets/", download=True,
                                                transform=data_transform, split='test')

        testloader = torch.utils.data.DataLoader(test, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        return trainloader, testloader
    
    if dataset_name == 'cifar10':

        data_transform = transforms.Compose(transform)

        train = torchvision.datasets.CIFAR10(root="./datasets/", 
                                             train=True,
                                             transform=data_transform,
                                             download=True)

        trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize,
                                                shuffle=True, num_workers=2)
        
        test = torchvision.datasets.CIFAR10(root="./datasets/", 
                                             train=False,
                                             transform=data_transform,
                                             download=True)
        
        testloader = torch.utils.data.DataLoader(test, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        return trainloader, testloader
        
    
    else:
        return None
    

def imshow(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(net, epochs, trainloader, criterion, optimizer, device, every_n_epochs=10):
    

    for i in range(10):    
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Device:", device)
    start_time = time.time()
    net = net.to(device)
    accuracy_hist = []
    
    print('######### Starting Training ########')
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_acc = 0.0

        for i, data in enumerate(tqdm(trainloader, 0)):
            inputs, labels = data
            optimizer.zero_grad()
            #------------------------------------------------------
            #------------------------------------------------------
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            max_scores, max_idx = outputs.max(dim=1)
            train_acc += torch.sum(max_idx == labels)/len(labels)


            #------------------------------------------------------
            #------------------------------------------------------
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            torch.cuda.empty_cache()

            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

        train_acc = train_acc.item()
        accuracy_hist.append(train_acc/(i+1))
        
        if epoch % every_n_epochs == 0:
            print(f"it:{epoch}/{epochs}, Average Accuracy:{train_acc/(i+1):.3f}")


    print('######### Finished Training ########')
    end_time = time.time()
    print('Total Trainig Time[s]: ', end_time - start_time, "\nAverage Training Time per Epoch [it/s]: ", (end_time-start_time)/epochs, "\nDevice:", device)


    plt.plot(accuracy_hist)
    plt.ylabel('Average Accuracy')
    plt.show()

    return net

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    


def create_directory(directory_path):
    ####################################################################################
    # Replace 'your_model_directory' with the desired path for your model directory
    # create_directory(model_directory)
    ####################################################################################

    try:
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")



def save_model(path, model, weights='model.pth'):
    torch.save(model.state_dict(), os.path.join(path,weights))


'''
WEIGHTS INITIALIZATION
'''

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                    
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 1)
                
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)


'''
MODEL BLOCKS
'''
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, input_features, output_features):
        super(residual_block, self).__init__()

        self.in_feat = input_features
        self.out_feat = output_features
                
        if input_features == output_features:
            self.Wi = [
                nn.Conv2d(input_features, output_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
            ]
            self.Ws = [
                nn.Identity()
            ]
        
        else:
            self.Wi = [
                nn.Conv2d(input_features, output_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
            ]
            self.Ws = [
                nn.Conv2d(input_features, output_features, kernel_size=1, stride=2, padding=0, bias=False)
            ]
        
        self.F = nn.Sequential(*self.Wi)
        self.I = nn.Sequential(*self.Ws)

    def forward(self, x):
        y = self.F(x) + self.I(x)  
        return y
    


'''
MODEL BUILDER
'''

def model_builder(model_definition):
    layers = []
    for layer in model_definition:
        if layer['type'] == 'residual':
            layers.append(
                residual_block(layer['input'],
                            layer['output']))
            
        if layer['type'] == 'conv':
            layers.append(
                nn.Conv2d(in_channels=layer['input'], 
                        out_channels=layer['output'], 
                        kernel_size=layer['kernel_size'], 
                        stride=layer['stride'], 
                        padding=layer['padding']))

        if layer['type'] == 'maxpool':
            layers.append(
                nn.MaxPool2d(2,2))

        if layer['type'] == 'adaptative':
            layers.append(
                nn.AdaptiveAvgPool2d((layer['output'])))

        if layer['type'] == 'flatt':
            layers.append(
                nn.Flatten(start_dim=layer['dim']))

        if layer['type'] == 'linear':
            layers.append(
                nn.Linear(layer['input'], layer['output']))
        
    return nn.Sequential(*layers)

