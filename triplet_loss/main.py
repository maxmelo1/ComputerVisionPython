import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch Triplet loss applied in MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='batch size for test')

    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../datasets/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../datasets', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    print(train_loader)

if __name__ == '__main__':
    main()