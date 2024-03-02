resnet = [
    {
        'type': 'conv', 
        'input': 3, 'output': 64,'kernel_size': 7,'stride': 2,'padding': 3
    },
    {
        'type': 'maxpool'
    },
    {
        'type': 'residual',
        'input': 64,'output': 128,
    },
    {
        'type': 'residual',
        'input': 128,'output': 256,
    },
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
        'input':1000, 'output':10
    }
]