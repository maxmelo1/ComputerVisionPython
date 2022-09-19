import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import scipy
from sklearn.model_selection import train_test_split
from tqdm import tqdm


IMAGE_DIR = 'cell_images/'
IMAGE_SIZE = 224

LEARNING_RATE = 1e-3
BS = 16
NUM_WORKERS = 2
PIN_MEMORY = True
NUM_EPOCHS = 10

dataset = np.array([])
label = np.array([])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(True)


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

class GAP(nn.Module):
    def global_average_polling_2d(self, x, keepims=False):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        if keepims:
            x = x.view(x.size(0), x.size(1), 1, 1)
        return x

    def __init__(self):
        super().__init__()

    def forward(self, x, keepdims=False):
        return self.global_average_polling_2d(x, keepdims)

class Model(nn.Module):
    def __init__(self, input_channels=3, n_classes = 2):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

        model_input = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        classifier_top = nn.Linear(in_features=4096, out_features=n_classes, bias=True)

        model.features[0] = model_input
        self.features = model.features
        # self.model.classifier[6] = classifier_top
        
        self.classifier = nn.Sequential(
            GAP(),
            nn.Linear(in_features=512, out_features=n_classes, bias=True)
        )

        

    def forward(self, x):
        x = self.features(x)
        
        return self.classifier(x)


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

parasitized_images = [IMAGE_DIR + 'Parasitized/'+el for el in os.listdir(IMAGE_DIR + 'Parasitized/') if '.png' in el]
l = np.ones(( len(parasitized_images)))
dataset = np.concatenate((dataset, parasitized_images))
label = np.concatenate((label, l))

uninfected_images = [IMAGE_DIR + 'Uninfected/'+el for el in os.listdir(IMAGE_DIR + 'Uninfected/') if '.png' in el]
l = np.zeros(( len(uninfected_images)))
dataset = np.concatenate((dataset, uninfected_images))
label = np.concatenate((label, l))


X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)


train_ds = CustomDatasetClassification(
    image_names= X_train,
    label_names= y_train,
    transform= train_transform,
)

val_ds = CustomDatasetClassification(
    image_names= X_test,
    label_names= y_test,
    transform= val_transform,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BS,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BS,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False
)

model = Model(n_classes=1)
# print( summary(model, (3, 224, 224)) )
# print(model)

_,x,y = next(iter(train_loader))
model.to(DEVICE)
x = x.to(DEVICE)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
scaler = torch.cuda.amp.GradScaler()

model.to(DEVICE)
model.train()
train_loss = 0.0
best_loss = 10000.
train_acc = 0.0
for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, unit="batch")
    for i, (im_name,x,y) in enumerate(loop):
        loop.set_description(f'Epoch: {epoch}, batch {i}')
        
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).unsqueeze(1).float()

        # with torch.cuda.amp.autocast():
        pred = model(x)
        loss = criterion(pred, y)
        train_loss += loss
        y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
        y_pred[pred>0.5] = 1.0
        acc = torch.sum(y_pred == y).cpu().detach().item() / BS
        train_acc += acc
        
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item(), acc=acc)

    train_loss = train_loss / (i+1)
    train_acc  = train_acc  / (i+1)
    print(f'On epoch end, train loss: {train_loss}, acc: {train_acc}')

    if train_loss < best_loss:
        print('New Best loss found, saving model')
        best_loss = train_loss
        model_path = 'best_model.pth'
        torch.save(model.state_dict(), model_path)

    print('evaluating model:')
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for i, (im_name, x, y) in enumerate(val_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1).float()

            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss
            
            y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
            y_pred[pred>0.5] = 1

            acc = torch.sum(y_pred == y).cpu().detach().item() / BS
            val_acc += acc

    val_loss = val_loss / (i+1)
    val_acc  = val_acc  / (i+1)
    print(f'Validation loss: {val_loss}, val acc: {val_acc}')
    
    scheduler.step()