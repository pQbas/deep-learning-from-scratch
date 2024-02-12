import os
import torch
import fnmatch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np
import pandas
from PIL import Image


transform = Compose([
    transforms.ToTensor(),  # Scales data into [0,1]
    transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
])


class CustomImageDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, data_transform, size):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.transf = data_transform
        self.size = size
        self.width = self.size[0]
        self.height = self.size[1]

    def __len__(self):
        return len(fnmatch.filter(os.listdir(self.img_dir), '*.*'))

    def __getitem__(self, idx):

        # Image ---------------------------------------------------
        img_path = os.path.join(self.img_dir, f'{idx}.png')
        image = Image.open(img_path).resize(self.size)
        #image = np.array(image).astype(np.float32)
        image = self.transf(image)
        
        dW = self.width//7
        dH = self.height//7

        # Boxes ---------------------------------------------------
        labels_path = os.path.join(self.annotations_dir, f'{idx}.csv')
        df = pandas.read_csv(labels_path)

        label = df['label'].to_numpy()
        xmin, xmax = df['xmin'].to_numpy(), df['xmax'].to_numpy()
        ymin, ymax = df['ymin'].to_numpy(), df['ymax'].to_numpy()

        factor_hor, factor_ver = self.width/300, self.height/300
        xmax, xmin = xmax*factor_hor, xmin*factor_hor
        ymax, ymin = ymax*factor_ver, ymin*factor_ver
        cx, cy = (xmin + xmax)//2, (ymin + ymax)//2
        px, py = (cx % dW)/dW, (cy % dH)/dH
        ph, pw = (xmax - xmin)/self.width, (ymax - ymin)/self.height

        boxes = np.stack(
                    (px, py, ph, pw),
                    axis= 1
                )

        # Probabilirt Class ---------------------------------------------
        n_objects = cx.shape[0]
        n_classes = 10

        class_prob = np.zeros((n_objects, n_classes))
        class_prob[np.arange(n_objects), label] = 1

        target = np.concatenate((boxes, class_prob), axis=1)


        # Choicing the bigger box if are solapamiento ---------------
        cell_i = np.uint8(cx//dW)
        cell_j = np.uint8(cy//dH)

        cell_position = np.zeros([n_objects, 7, 7, 14])
        cell_position[np.arange(n_objects), cell_i, cell_j, :] = 1

        cell_position_sum = cell_position.sum(axis=0)
        i_x, i_y = np.where(cell_position_sum[..., 0] > 1)

        k_idx = []
        for ix, iy in zip(i_x, i_y):
            k = np.where(cell_i == ix, 1, 0) * np.where(cell_j == iy, 1, 0)
            k_idx.append(np.where(k == 1)[0])

        if ~(len(k_idx) == 0):     
            vect_ = np.ones_like(target)
            for boxes in k_idx:
                areas = []

                for box in boxes:
                    wn, hn = target[box, 2:4]
                    areas.append(wn*hn)

                max_index = np.argmax(np.array(areas))
                
                for k, box in enumerate(boxes):
                    if k == max_index:
                        continue
                    vect_[box, :] = 0

            target_corrected = target*vect_
            cell_position_corrected = cell_position * vect_[:, None, None, :]

            target_corrected = target_corrected[:, None, None, :] * cell_position_corrected
            target_corrected = target_corrected.sum(axis=0)

            cell_position_corrected_sum = cell_position_corrected.sum(axis=0)

            obj_position_ = cell_position_corrected_sum[...,0]
            target_ = target_corrected

            assert target_.shape == (7, 7, 14)
            assert obj_position_.shape == (7, 7)

            no_obj_position = 1 - obj_position_
            no_obj_position = no_obj_position[...][..., None]

            target_out = np.concatenate((target_, no_obj_position), axis=2)
            target_out = np.concatenate((target_out, obj_position_[...][..., None]), axis=2)

            target = target_out

        target = torch.tensor(target.astype(np.float32))

        return image, {
            'bbox': target[..., 0:4],
            'class': target[..., 4:15],
            'one_obj': target[..., 15:16]
        }
    

if __name__ == '__main__':
    dataset = CustomImageDataset(annotations_dir = '/home/pqbas/dl/detection/MNIST-ObjectDetection/data/mnist_detection/test/labels',
                                img_dir = '/home/pqbas/dl/detection/MNIST-ObjectDetection/data/mnist_detection/test/images',
                                data_transform = transform,
                                size=(448,448))

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)


    ####################################################
    #############         Testing         ##############
    ####################################################

    break_n = 1

    for idx, (image, target) in enumerate(train_loader):

        # print(target['bbox'])
        # print(target['class'])
        print(target['one_obj'].shape)
        if idx == break_n:
            break