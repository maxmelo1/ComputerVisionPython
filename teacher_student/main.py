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


def main():
    parser = argparse.ArgumentParser(description='Knowledge distillation for semantic segmentation')
    parser.add_argument("--img_dir", type=str, default='./data/VOCdevkit/VOC2012/JPEGImages/')
    parser.add_argument("--train_list", type=str, default='./data/voc12/train_cls.txt')
    parser.add_argument("--test_list", type=str, default='./data/voc12/val_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
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

    args = parser.parse_args()

    

if __name__ == '__main__':
    main()