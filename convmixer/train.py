import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from model2 import ConvMixer

from tqdm import tqdm


def train(model, loss_fn, optim, dl_train, dl_val, device, num_epochs):

    device = torch.device(device)
    model.to(device)

    print(torch.cuda.get_device_properties(0))

    avg_loss = []
    val_loss = []
    lowest_loss = 100000


    for epoch in range(num_epochs):
        
        loss = []
        with tqdm(total=len(dl_train), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            pbar.set_description(f'Epoch {epoch}/{num_epochs}')
            model.train()
            for X,y in dl_train:
                X = X.to(device)
                y = y.to(device)

                optim.zero_grad()

                pred = model(X)
                l = loss_fn(pred, y)

                l.backward()
                optim.step()

                loss.append(l)
                pbar.update(1)
                pbar.set_postfix(loss=l.cpu().item())
                
        l = torch.mean(torch.stack(loss))
        avg_loss.append(l.cpu().item())
        print("[%d/%d] - epoch end loss: %f"%(epoch,num_epochs,avg_loss[-1]))

        

        

        
        model.eval()
        with torch.no_grad():
            loss = []
            lacc = []
            with tqdm(total=len(dl_val), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
                pbar.set_description(f'Validating epoch {epoch}/{num_epochs}')
                for x, y in dl_val:
                    x = x.to(device)
                    y = y.to(device)
                    preds = model(x)
                    
                    l = loss_fn(preds, y)
                    loss.append(l)

                    _, preds = preds.max(1)
                    total = y.size(0)

                    num_correct = preds.eq(y).sum()

                    # print(preds)
                    # print('----')
                    # print(y)
                    # input()
                    
                    acc = num_correct/total

                    lacc.append(acc)

                    pbar.update(1)

                    pbar.set_postfix(acc=acc.item())

            avg_acc         = torch.mean(torch.stack(lacc))
            avg_val_loss    = torch.mean(torch.stack(loss))
            print(f'Val acc: {avg_acc}, Val loss: {avg_val_loss}')



transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
    ])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
# show images
# imshow(torchvision.utils.make_grid(images))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model       = ConvMixer(128, filters=256, depth=8, kernel_size=7, patch_size=7, num_classes=10).to(device)
model       = ConvMixer(128, depth=8, kernel_size=7, patch_size=7, n_classes=10).to(device)
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)




train(model, criterion, optimizer, trainloader, testloader, device, 30)