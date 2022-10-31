import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model, GAP
from customdatasetclassification import CustomDatasetClassification
from params import *

from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import numpy as np

model_path = 'best_model.pth'

model = Model(n_classes=1)
model.load_state_dict(torch.load(model_path))

val_transform = A.Compose(
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

dataset = np.array([])
label = np.array([])

parasitized_images = [IMAGE_DIR + 'Parasitized/'+el for el in os.listdir(IMAGE_DIR + 'Parasitized/') if '.png' in el]
l = np.ones(( len(parasitized_images)))
dataset = np.concatenate((dataset, parasitized_images))
label = np.concatenate((label, l))

uninfected_images = [IMAGE_DIR + 'Uninfected/'+el for el in os.listdir(IMAGE_DIR + 'Uninfected/') if '.png' in el]
l = np.zeros(( len(uninfected_images)))
dataset = np.concatenate((dataset, uninfected_images))
label = np.concatenate((label, l))


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)


val_ds = CustomDatasetClassification(
    image_names= X_test,
    label_names= y_test,
    transform= val_transform,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BS,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False
)

model.to(DEVICE)
model.eval()

macc = []
with torch.no_grad():
    for i, (im_name,x,y) in enumerate(val_loader):
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).unsqueeze(1).float()

        pred = model(x)
        y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
        y_pred[pred>0.5] = 1.0

        # print(y_pred)
        # print(y)

        acc = torch.sum(y_pred == y) / BS
        # print()

        print(acc.cpu().item())
        macc.append(acc.cpu().item())

print(f'Mean acc: {np.mean(macc)}')