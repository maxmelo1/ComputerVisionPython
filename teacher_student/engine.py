import torch
import matplotlib.pyplot as plt
from dataset import denormalize

def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch, args):
    model.train()

    for images, labels in data_loader:

        plt.imshow(denormalize(images[0]).permute(1, 2, 0).cpu().numpy())
        plt.show()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()