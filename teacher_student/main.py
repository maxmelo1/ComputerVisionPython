# based on: https://github.com/PengtaoJiang/L2G

import sys
import os

import torch
import argparse
import time

from tqdm import trange, tqdm
import importlib
import shutil

import torch.optim as optim
import torch.nn.functional as F

from models.networks import Classifier
from dataset import VOCDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from engine import train_one_epoch


def main():
    parser = argparse.ArgumentParser(description='Knowledge distillation for semantic segmentation')
    parser.add_argument("--img_dir", type=str, default='./data/VOCdevkit/VOC2012/JPEGImages/')
    parser.add_argument("--base_dir", type=str, default='/mnt/7B887C703B06DD07/datasets/new_voc_12/new_voc_12/VOC2012')
    parser.add_argument("--train_list", type=str, default='./data/voc12/train.txt')
    parser.add_argument("--test_list", type=str, default='./data/voc12/val.txt')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iter_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--poly_optimizer", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default='first_model')
    parser.add_argument("--model", type=str, default='resnet50')

    args = parser.parse_args()

    print('Parameters:\n', args)

    if not os.path.exists(args.experiment_name):
        os.makedirs(args.experiment_name)

    model = Classifier(args.model, args.num_classes, 'normal')
    model = model.cuda()

    model_local = Classifier(args.model, args.num_classes, 'normal')
    model_local = model_local.cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_local.to(device)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


    dataset = VOCDataset(args.base_dir, args.train_list, args.crop_size, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = torch.nn.MSELoss()

    for epoch in range(args.epoch):
        train_one_epoch(model, optimizer, criterion, dataloader, device, epoch, args)

if __name__ == '__main__':
    main()