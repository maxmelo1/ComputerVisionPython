import torch
import torch.nn as nn



class GAP(nn.Module):
    def global_average_polling_2d(self, x, keepims=False):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        if keepims:
            x = x.view(x.size(0), x.size(1), 1, 1)
        return x

    def __init__(self):
        super().__init__()

    def forward(self, x, keepdims=False):
        return self.global_average_polling_2d(x, keepdims)

class Model(nn.Module):
    def __init__(self, input_channels=3, n_classes = 2):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

        model_input = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # classifier_top = nn.Linear(in_features=4096, out_features=n_classes, bias=True)

        model.features[0] = model_input
        self.features = model.features
        # self.model.classifier[6] = classifier_top
        
        self.classifier = nn.Sequential(
            GAP(),
            nn.Linear(in_features=512, out_features=n_classes, bias=True),
            # nn.Softmax(dim=1),
        )

        

    def forward(self, x):
        x = self.features(x)
        
        return self.classifier(x)
