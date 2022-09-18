import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import scipy
from sklearn.model_selection import train_test_split


IMAGE_DIR = 'cell_images/'
IMAGE_SIZE = 224

dataset = np.array([])
label = np.array([])


class CustomDatasetClassification(Dataset):
    def __init__(self, image_names, label_names, transform=None):
        self.images = image_names
        self.labels = label_names
         
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        label = self.labels[index]

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return self.images[index], image, label



train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

parasitized_images = [IMAGE_DIR + 'Parasitized/'+el for el in os.listdir(IMAGE_DIR + 'Parasitized/')]
l = np.ones(( len(parasitized_images)))
dataset = np.concatenate((dataset, parasitized_images))
label = np.concatenate((label, l))

uninfected_images = [IMAGE_DIR + 'Uninfected/'+el for el in os.listdir(IMAGE_DIR + 'Uninfected/')]
l = np.zeros(( len(uninfected_images)))
dataset = np.concatenate((dataset, uninfected_images))
label = np.concatenate((label, l))


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)


train_ds = CustomDatasetClassification(
    image_names= X_train,
    label_names= y_train,
    transform= train_transform,
)

print(len(train_ds))

im_name, img, lbl = next(iter(train_ds))
print(im_name)
plt.figure()
plt.imshow(img.cpu().detach().numpy().transpose(1,2,0))
plt.show()