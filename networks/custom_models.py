import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision.models.resnet import ResNet50_Weights
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=True):
        super(EfficientNet, self).__init__()
    
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        num_ftrs = self.model.classifier[1].in_features

        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            
            self.model.classifier[1] = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),  # Add ReLU activation if needed
                nn.Linear(intermediate_dim, num_classes)  # Output layer
            )
            
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.classifier[1][2].weight.data, 0.0, init_gain)
        else:
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.classifier[1].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=True):
        super(VGG16, self).__init__()
        
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)

        num_ftrs = self.model.classifier[6].in_features
        
        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            
            self.model.classifier[6] = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),  # Add ReLU activation if needed
                nn.Linear(intermediate_dim, num_classes)  # Output layer
            )
            
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.classifier[6][2].weight.data, 0.0, init_gain)
        else:
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.classifier[6].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)
    

class ResNet50(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=True):
        super(ResNet50, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Get the number of input features for the final fully connected layer
        num_ftrs = self.model.fc.in_features
        
        # Define the intermediate layer if requested
        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            
            # Replace the final fully connected layer with a sequence of layers
            # containing the intermediate layer and the final output layer
            self.model.fc = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),  # Add ReLU activation if needed
                nn.Linear(intermediate_dim, num_classes)  # Output layer
            )
            
            # Initialize the weights of the new layers
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.fc[2].weight.data, 0.0, init_gain)
        else:
            # If not adding the intermediate layer, replace the final fully connected layer directly
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)