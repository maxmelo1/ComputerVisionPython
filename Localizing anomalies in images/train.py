import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model import Model, GAP
from customdatasetclassification import CustomDatasetClassification
from params import *


dataset = np.array([])
label = np.array([])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# torch.autograd.set_detect_anomaly(True)




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


model = Model(n_classes=2)
print(model)

_,x,y = next(iter(train_loader))
model.to(DEVICE)
x = x.to(DEVICE)

print( summary(model, (3, 224, 224)) )


ct = 0
for child in model.features.children():
    if ct < 17:
        # print(child._get_name())
        for param in child.parameters():
            param.requires_grad = False
    ct += 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
scaler = torch.cuda.amp.GradScaler()

model.to(DEVICE)
model.train()
train_loss = 0.0
best_loss = 10000.
train_acc = 0.0

train_log = {'loss' : [], 'acc': []}
val_log = {'loss' : [], 'acc': []}

for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, unit="batch")
    for i, (im_name,x,y) in enumerate(loop):
        loop.set_description(f'Epoch: {epoch}, batch {i}')
        
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).long()#.unsqueeze(0)#.float()

        # with torch.cuda.amp.autocast():
        pred = model(x)
        # pred = torch.argmax(pred, dim=1)
        
        loss = criterion(pred, y)
        train_loss += loss
        # y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
        # y_pred[pred>0.5] = 1.0
        y_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()
        acc = torch.sum(y_pred == y).cpu().detach().item() / BS
        train_acc += acc
        
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loop.set_postfix(loss=loss.item(), acc=acc)
        loop.set_postfix(loss= train_loss.item() / (i+1), acc= train_acc / (i+1))

    train_loss = train_loss / (i+1)
    train_acc  = train_acc  / (i+1)
    print(f'On epoch end, train loss: {train_loss}, acc: {train_acc}')
    train_log['loss'].append(train_loss.cpu().detach())
    train_log['acc'].append(train_acc)

    print('evaluating model:')
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for i, (im_name, x, y) in enumerate(val_loader):
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).long()#.unsqueeze(1).float()

            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss
            
            # y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
            # y_pred[pred>0.5] = 1
            y_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()

            acc = torch.sum(y_pred == y).cpu().detach().item() / BS
            val_acc += acc

    val_loss = val_loss / (i+1)
    val_acc  = val_acc  / (i+1)
    print(f'Validation loss: {val_loss}, val acc: {val_acc}')
    val_log['loss'].append(val_loss.cpu().detach())
    val_log['acc'].append(val_acc)

    if val_log['loss'][-1] < best_loss:
        best_loss = val_log['loss'][-1]
        print(f'New Best loss found: {best_loss}. Saving model')
        model_path = 'best_model.pth'
        torch.save(model.state_dict(), model_path)
    
    scheduler.step()

#plotar os gráficos
epochs = range(1, NUM_EPOCHS+1)
plt.plot(epochs, train_log['loss'], 'y', label='Training loss')
plt.plot(epochs, val_log['loss'], 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

plt.plot(epochs, train_log['acc'], 'y', label='Training acc')
plt.plot(epochs, val_log['acc'], 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.png')
plt.show()