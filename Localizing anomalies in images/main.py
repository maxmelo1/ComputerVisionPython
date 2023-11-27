import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision

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
from model import Model, CAM
from customdatasetclassification import CustomDatasetClassification, inv_z_score
from params import *
import argparse

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

import seaborn as sns
import shutil


dataset = np.array([])
label = np.array([])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# torch.autograd.set_detect_anomaly(True)

def str2bool(key):
    if isinstance(key, bool):
        return key
    if key.lower() in ['true', 't', 'v']:
        return True
    return False

parser = argparse.ArgumentParser('GAP example')

parser.add_argument('--mode', default='train', choices=['train', 'eval'])
parser.add_argument('--image_dir', default='cell_images/')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--image_size', default=224)
parser.add_argument('--learning_rate', default=1e-3)
parser.add_argument('--visualize', default=False, type=str2bool)

args = parser.parse_args()

IMAGE_DIR       = args.image_dir
NUM_EPOCHS      = args.num_epochs
BS              = args.batch_size
IMAGE_SIZE      = args.image_size
LEARNING_RATE   = args.learning_rate


mean    = np.array([0.52868627, 0.4217098 , 0.44797647])
std     = np.array([0.34581961, 0.27895686, 0.29270588])

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


with open('./data/train.txt', 'r') as f:
    X_train = np.array(f.read().splitlines())
with open('./data/train_labels.txt', 'r') as f:
    y_train = np.array(f.read().splitlines())

with open('./data/val.txt', 'r') as f:
    X_test = np.array(f.read().splitlines())
with open('./data/val_labels.txt', 'r') as f:
    y_test = np.array(f.read().splitlines())




# X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)


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


def evaluate(loader, model, criterion):
    model.eval()
    sum_loss = 0.0
    sum_acc  = 0.0
    with torch.no_grad():
        for i, (im_name, x, y) in enumerate(loader):
            x = x.to(DEVICE)#.float()
            y = y.to(DEVICE)#.unsqueeze(1)#.float()

            pred = model(x)
            loss = criterion(pred, y)
            sum_loss += loss

            # y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
            # y_pred[pred>0.5] = 1
            y_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()
            gt     = torch.argmax(y, dim=1).long()
            # y_pred = (nn.Sigmoid()(pred)> 0.5).float()

            acc = torch.sum(y_pred == gt).cpu().detach().item() / BS
            sum_acc += acc

    sum_loss = sum_loss / (i+1)
    sum_acc  = sum_acc  / (i+1)
    return sum_loss, sum_acc





# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()

model_path = 'best_model.pth'

train_loss = 0.0
best_loss = 10000.
train_acc = 0.0

train_log = {'loss' : [], 'acc': []}
val_log = {'loss' : [], 'acc': []}


if args.mode == 'train':
    model = Model(n_classes=2)
    print(model)

    ct = 0
    for child in model.features.children():
        if ct < 17:
            # print(child._get_name())
            for param in child.parameters():
                param.requires_grad = False
        ct += 1

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    _,x,y = next(iter(train_loader))
    model.to(DEVICE)
    x = x.to(DEVICE)

    summary(model, (3, 224, 224))

    model.to(DEVICE)
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, unit="batch")
        for i, (im_name,x,y) in enumerate(loop):
            loop.set_description(f'Epoch: {epoch}, batch {i}')

            x = x.to(DEVICE)#.float()
            y = y.to(DEVICE)#.unsqueeze(-1)#.long()#.unsqueeze(0)#.float()


            # with torch.cuda.amp.autocast():
            pred = model(x)
            # pred = torch.argmax(pred, dim=1)

            # print(pred.type(), y.type())
            # print(pred.size())
            # print(y.size())
            # input()

            loss = criterion(pred, y)
            train_loss += loss
            # y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
            # y_pred[pred>0.5] = 1.0
            y_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()
            gt = torch.argmax(y, dim=1).long()

            # y_pred = (nn.Sigmoid()(pred)> 0.5).float()
            acc = torch.sum(y_pred == gt).cpu().detach().item() / BS
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
        

        val_loss, val_acc = evaluate(val_loader, model=model, criterion=criterion)
        
        print(f'Validation loss: {val_loss}, val acc: {val_acc}')
        val_log['loss'].append(val_loss.cpu().detach())
        val_log['acc'].append(val_acc)

        if val_log['loss'][-1] < best_loss:
            best_loss = val_log['loss'][-1]
            print(f'New Best loss found: {best_loss}. Saving model')
            
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

elif args.mode == 'eval':
    model = Model(n_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    base_model = nn.Sequential(*list(model.children())[:-1][0][:-2])
    base_model.to(DEVICE)

    
    print(model)
    summary(model, (3, 224, 224))
    print('Model successfully loaded from disk!')



    test_loss, test_acc = evaluate(loader=val_loader, model=model, criterion=criterion)

    print(f'Val loss: {test_loss}, val acc: {test_acc}')

    val_gt      = []
    val_preds   = []


    if args.visualize == True:
        if os.path.exists(os.path.join('./results/CAM/')):
            print('Deleting previous generated CAMs')
            shutil.rmtree('./results/CAM/')
            print('Done!')

        os.makedirs(os.path.join('./results/CAM/'), exist_ok=True)

        print(f'{len(val_ds)} validation images. Generating CAMs...')
        
        #used to save the last batch 
        for i, (_, x, y) in enumerate(val_loader):
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).long()
            pred = model(x)
            y_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()
            # y_pred = (nn.Sigmoid()(pred)> 0.5).float()

            # if sum(torch.argmax(y, dim=1)) > 0:                

            probs = F.softmax(pred, dim=1).data.squeeze()
            
            # probs, idx = h_x.sort(0, True)
            idx = torch.argmax(probs, dim=1)
            # print('-->', idx)

            probs = probs.detach().cpu().numpy()
            idx = idx.cpu().numpy()

            params = list(model.parameters())
            weight = np.squeeze(params[-1].data.cpu().numpy())

            features_blobs = base_model(x)
            features_blobs1 = features_blobs.cpu().detach().numpy()

            CAMs = CAM(features_blobs1, weight, [idx[0]], size=(IMAGE_SIZE, IMAGE_SIZE))[0]

            # print(np.shape(CAMs), np.shape(x))
            if i < len(val_loader):
               continue

            for j, (img, cam) in enumerate(zip(x, CAMs)):
                heatmap = cv2.applyColorMap(cv2.resize(cam,(IMAGE_SIZE, IMAGE_SIZE)), cv2.COLORMAP_JET)
                img = img.permute(1,2,0).detach().cpu().numpy()
                img = inv_z_score(img, mean, std )
                result = heatmap * 0.5 + img * 0.5

                # plt.imshow(result[..., ::-1])
                # plt.show()

                cv2.imwrite(os.path.join('./results/CAM/', f'img_{i}.png' ), img)
                cv2.imwrite(os.path.join('./results/CAM/', f'cam_{i}.png' ), result)


            # print(y)
            val_gt      += y.cpu().detach().tolist()
            val_preds   += y_pred.cpu().detach().tolist()

    print('Done!')

    val_gt = torch.argmax(torch.as_tensor(val_gt), dim=1).tolist()
    
    #print(val_preds)

    cm_GAP=confusion_matrix(val_gt, val_preds)
    print(cm_GAP)
    cmatrix = sns.heatmap(cm_GAP, annot=True)

    cmatrix.figure.savefig("output.png")
    plt.show()

    roc_auc = roc_auc_score(val_gt, val_preds)
    fpr, tpr, thresholds = roc_curve(val_gt, val_preds)

    fig = plt.figure() 
    plt.plot(fpr,tpr)
    plt.savefig('roc_curve.png')
    plt.show()

    print(roc_auc)
    
