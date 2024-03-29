import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        torch.nn.init.normal_(self.model.classifier[1].weight.data, 0.0, init_gain)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, nb_intermediate=50):
        super(VGG16, self).__init__()
        
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)

        num_features = self.model.classifier[6].in_features  # Usually 4096 for VGG16
        self.model.classifier[6] = nn.Linear(num_features, nb_intermediate)
        torch.nn.init.normal_(self.model.classifier[6].weight.data, 0.0, init_gain)
        
        self.final_fc = nn.Linear(nb_intermediate, num_classes)
        torch.nn.init.normal_(self.final_fc.weight.data, 0.0, init_gain)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.final_fc(x)
        x = self.softmax(dim=1)
        return x