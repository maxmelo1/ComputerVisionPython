import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from model import Model, GAP
from customdatasetclassification import CustomDatasetClassification
from params import *

from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import seaborn as sns

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle #To add a rectangle overlay to the image
from skimage.feature.peak import peak_local_max  #To detect hotspots in 2D images. 
import scipy

model_path = 'best_model.pth'

model = Model(n_classes=1)
model.load_state_dict(torch.load(model_path))

# print( summary(model, (3, 224, 224)) )
print(model)

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

y_pred_list = []
y_list = []


with torch.no_grad():
    for i, (im_name,x,y) in enumerate(val_loader):
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).unsqueeze(1).float()

        pred = model(x)
        y_pred = torch.zeros(pred.size(), dtype=torch.float32).to(DEVICE)
        y_pred[pred>0.5] = 1.0

        y_list += y.flatten().cpu().tolist()
        y_pred_list += y_pred.flatten().cpu().tolist()

        # print(y_pred.cpu().flatten().tolist())
        # print(y)
        # input()

        acc = torch.sum(y_pred == y) / BS
        # print()

        print(acc.cpu().item())
        macc.append(acc.cpu().item())

print(f'Mean acc: {np.mean(macc)}')

i = random.randint(0, len(val_loader)-1)
j = random.randint(0, BS-1)

# print(len(y_pred_list))
# print(i*BS+j, i ,j, len(val_loader), len(val_ds))

yh = y_pred_list[i*BS+j]
y = y_list[i*BS+j]


fig, ax = plt.subplots(1)
ax.imshow(Image.open(val_ds[i*BS+j][0]))
ax.text(5, 5, f'Predicted class: {yh}, GT: {y}')
plt.savefig('sample_check.png')
plt.show()

print(f'Predicted class: {yh}, GT: {y}')

cm=confusion_matrix(y_list, y_pred_list)  
sns.heatmap(cm, annot=True)
plt.show()


def plot_heatmap(img):
 
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(np.expand_dims(val_transform(image=img)['image'], axis=0)).to(DEVICE)
        print(img.size())

    pred = model(img)
    pred_class = pred.flatten()
    #Get weights for all classes from the prediction layer
    last_layer_weights = model.classifier[-1].weight#[0] #Prediction layer
    print(last_layer_weights)
    print(last_layer_weights.size())
    #Get weights for the predicted class.
    last_layer_weights_for_pred = last_layer_weights[:, pred_class]
    #Get output from the last conv. layer
    last_conv_model = Model(model.input, model.get_layer("block5_conv3").output)
    last_conv_output = last_conv_model(img[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    
    #Upsample/resize the last conv. output to same size as original image
    h = int(img.shape[0]/last_conv_output.shape[0])
    w = int(img.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 512)), 
                 last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
    
    #Since we have a lot of dark pixels where the edges may be thought of as 
    #high anomaly, let us drop all heat map values in this region to 0.
    #This is an optional step based on the image. 
    heat_map[img[:,:,0] == 0] = 0  #All dark pixels outside the object set to 0
    
    #Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
    #with rel threshold of 0.5 (compared to the max peak). 
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10) 

    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
    plt.imshow(heat_map, cmap='jet', alpha=0.30)
    for i in range(0,peak_coords.shape[0]):
        print(i)
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        plt.gca().add_patch(Rectangle((x-25, y-25), 50,50,linewidth=1,edgecolor='r',facecolor='none'))

while True:
    if val_ds[i*BS+j][2] and yh[i*BS+j]:
        cl = val_ds[i*BS+j][2]
        im_name = val_ds[i*BS+j][0]
        print(f'image: {im_name} class: {cl}')
        break
    i = random.randint(0, len(val_loader)-1)
    j = random.randint(0, BS-1)



heat_map = plot_heatmap(np.array(Image.open(im_name)))
