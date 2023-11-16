import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


def one_hot_encoding(y, n_classes):
    labels = np.zeros((n_classes), dtype=np.float32)

    labels[y.astype(int)] = 1
    return labels

class CustomDatasetClassification(Dataset):
    def __init__(self, image_names, label_names, transform=None, num_classes=2):
        self.images = image_names
        self.labels = label_names
         
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        
        label = one_hot_encoding(self.labels[index], n_classes=self.num_classes)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return self.images[index], image, label