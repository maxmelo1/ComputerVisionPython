import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

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