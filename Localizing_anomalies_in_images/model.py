import torch
import torch.nn as nn

import numpy as np
import cv2


class GAP(nn.Module):
    def global_average_polling_2d(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        if self.keepdims:
            x = x.view(x.size(0), x.size(1), 1, 1)
        return x

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x):
        return self.global_average_polling_2d(x)

class Model(nn.Module):
    def __init__(self, input_channels=3, n_classes = 2):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights='VGG16_Weights.DEFAULT')

        model_input = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # classifier_top = nn.Linear(in_features=4096, out_features=n_classes, bias=True)

        model.features[0] = model_input
        self.features = model.features
        # self.model.classifier[6] = classifier_top
        
        self.classifier = nn.Sequential(
            GAP(keepdims=False),
            nn.Linear(in_features=512, out_features=n_classes, bias=False),
            # nn.Softmax(dim=1),
        )

        

    def forward(self, x):
        x = self.features(x)
        
        return self.classifier(x)




def CAM(features, weight, class_idx, size):
    bs, c, h, w = features.shape
    
    output_cam = []

    for idx in class_idx:
        # print('weeeight', weight.shape)
        # print('features', features.shape, idx)
        # input()

        beforeDot =  features.reshape((bs, c, h*w))
        
        cam = np.matmul(weight[idx], beforeDot)
        
        cam = cam.reshape(bs, h, w)
        cam = np.expand_dims(cam, axis=3)


        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        cam_img = np.uint8(255 * cam_img)
        # output_cam.append(cv2.resize(cam_img, size))

        cam_imgs = list(map(lambda x: cv2.resize(x, size), cam_img))
        output_cam.append(cam_imgs)

    # print(np.shape(output_cam[0]))
    # print(output_cam[0])
    # input()
    
    return output_cam