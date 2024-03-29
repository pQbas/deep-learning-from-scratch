import torch
from torch import nn


class YOLOv1(nn.Module):
    def __init__(self, S, depth):
        super().__init__()
        self.depth = depth
        self.S = S
        self.adaptative = nn.AdaptiveAvgPool2d((1))

        layers = [
            # Probe(0, forward=lambda x: print('#' * 5 + ' Start ' + '#' * 5)),
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),                   # Conv 1
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv1', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),                           # Conv 2
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv2', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),                                     # Conv 3
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv3', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(1):                                                          # Conv 4
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv4', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(2):                                                          # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv5', forward=probe_dist),
        ]

        for _ in range(2):                                                          # Conv 6
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        # layers.append(Probe('conv6', forward=probe_dist))

        layers += [
            nn.AdaptiveAvgPool2d((1)),
            nn.Flatten(),
            #nn.Linear(S * S * 1024, 4096),                            # Linear 1
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('linear1', forward=probe_dist),
            #nn.Linear(4096, S * S * self.depth),                      # Linear 2
            nn.Linear(1024, S * S * self.depth)
            # Probe('linear2', forward=probe_dist),
        ]

        self.model = nn.Sequential(*layers)
        self.float()

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), self.S, self.S, self.depth)
        )

resnet = [
    {'type': 'conv', 
     'input': 1, 'output': 64,'kernel_size': 7,'stride': 2,'padding': 3},
    {'type': 'maxpool'},
    {'type': 'residual','input': 64,'output': 128,},
    {'type': 'residual','input': 128,'output': 256,},
    {
        'type': 'residual',
        'input': 256,'output': 512,
    },
    {
        'type': 'residual',
        'input': 512,'output': 1000,
    },
    {
        'type':'adaptative',
        'output': 1
    },
    {
        'type': 'flatt',
        'dim': 1
    },
    {
        'type':'linear',
        'input':1000, 'output':7*7*12
    }
]

import torch
from torch import nn
from utils import model_builder

class YOLOv1_resnet(nn.Module):
    def __init__(self, depth, S):
        super().__init__()
        self.depth = depth
        self.S = S
        self.backbone = model_builder(resnet)


    def forward(self, x):
        return torch.reshape(
            self.backbone.forward(x),
            (x.size(dim=0), self.S, self.S, self.depth)
        )




if __name__ == '__name__':
    ###############################
    #         TESTING             #
    ###############################


    model = YOLOv1(2, 10, 7)
    x = torch.rand((1, 1, 448, 448))
    pred = model(x)

    print(pred.shape)
