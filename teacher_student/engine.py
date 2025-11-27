import torch
import matplotlib.pyplot as plt
from dataset import denormalize
from utils.averageMeter import AverageMeter

def train_one_epoch(model_global, model_local, optimizer_global, optimizer_local, criterion_global, criterion_local, data_loader, device, epoch, args):
    model_global.train()
    model_local.train()

    losses_global = AverageMeter()
    losses_local = AverageMeter()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.long().squeeze(1).to(device)

        optimizer_global.zero_grad()
        outputs_global = model_global(images)
        # outputs_local = model_local(images)
        
        loss_global = criterion_global(outputs_global, labels)
        loss_global.backward()
        optimizer_global.step()

        # data_loader.set_postfix({
        #     'Global Loss': f'{losses_global.avg:.4f}',
        #     # 'Local Loss': f'{losses_local.avg:.4f}'
        # })

    print(f'\nEpoch {epoch + 1} - Global Loss: {losses_global.avg:.4f}')  # Add Local Loss when ready
    return losses_global.avg  # Return the average loss for this epoch