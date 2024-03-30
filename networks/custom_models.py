import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights

# class EfficientNet(nn.Module):
#     def __init__(self, num_classes=1, init_gain=0.02):
#         super(EfficientNet, self).__init__()
#         self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
#         num_ftrs = self.model.classifier[1].in_features
#         self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
#         torch.nn.init.normal_(self.model.classifier[1].weight.data, 0.0, init_gain)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         return self.model(x)
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64):
        super(EfficientNet, self).__init__()
    
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        num_ftrs = self.model.classifier[1].in_features
        self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)

        self.model.classifier[1] = nn.Sequential(
            self.intermediate_layer,
            nn.ReLU(inplace=True),  # Add ReLU activation if needed
            nn.Linear(intermediate_dim, num_classes)  # Output layer
        )

        torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
        torch.nn.init.normal_(self.model.classifier[1][2].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64):
        super(VGG16, self).__init__()
        
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)

        self.intermediate_layer = nn.Linear(4096, intermediate_dim)

        self.model.classifier[6] = nn.Sequential(
            self.intermediate_layer,
            nn.ReLU(inplace=True),  # Add ReLU activation if needed
            nn.Linear(intermediate_dim, num_classes)  # Output layer
        )

        # Initialize the weights of the new layers
        torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(self.model.classifier[6][2].weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.model(x)