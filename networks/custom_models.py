import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B4_Weights
from torchvision.models.resnet import ResNet50_Weights
from transformers import AutoModel, AutoConfig
import timm


class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # assuming avg pooling is needed
        self.flatten = nn.Flatten()  # to flatten the pooled output
        self.drop = nn.Dropout(p=0.0)  # adjust dropout probability if needed
        self.fc = nn.Linear(in_features, num_classes)  # adjust num_classes as needed

    def forward(self, x, pre_logits=None):
        # Check if pre_logits is callable and use it if so
        if callable(pre_logits):
            x = pre_logits(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
            
            
def load_custom_model(name: str, intermediate, intermediate_dim):

    model = None

    if name == 'res50':
        model = ResNet50(add_intermediate_layer = intermediate, intermediate_dim=intermediate_dim)
    elif name == 'vgg16':
        model = VGG16(add_intermediate_layer = intermediate, intermediate_dim=intermediate_dim)
    elif name == 'efficient_b0':
        model = EfficientNet_b0(add_intermediate_layer = intermediate, intermediate_dim=intermediate_dim)
    elif name == 'efficient_b4':
        model = EfficientNet_b4(add_intermediate_layer = intermediate, intermediate_dim=intermediate_dim)
    elif name == 'swin_tiny':
        model = HuggingModel("microsoft/swin-tiny-patch4-window7-224", 1) #["base_model.encoder.layers.3.blocks.1"]
    elif name == 'swin_base':
        model = HuggingModel("microsoft/swin-base-patch4-window7-224", 1)
    elif name == 'swin_large':
        model = HuggingModel("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft", 1)
    elif name == "coatnet":
        # Assuming you're using the model 'coatnet_0_rw_224.sw_in1k' and want NUM_CLASSES as output
        model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True)
        NUM_CLASSES = 1  # Set the number of classes as per your dataset requirement
        # Replace the 'head' with your new custom head
        model.head = CustomHead(in_features=768, num_classes=NUM_CLASSES)
        model = model
    elif name == "resnext":
        model = timm.create_model('resnext101_32x8d', pretrained=True)
        # Assuming the last layer was named 'head' and you verified it has an attribute 'in_features'
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, model.fc.in_features),
            nn.ReLU(),
            nn.Linear(model.fc.in_features, 1)
        )
    else:
        raise ValueError("Model name should either be res50, vgg16, efficient_b0, or efficient_b4")
    return model


class HuggingModel(nn.Module):
    def __init__(self, base_mod_name, NUM_CLASSES=1, freeze_layers=None, additional_layers=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_mod_name)
        self.base_model = AutoModel.from_pretrained(base_mod_name, config=self.config)
        
        if freeze_layers:
            for name, param in self.base_model.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
        
        if additional_layers:
            print('Additional')
            self.classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, NUM_CLASSES)
            )
        else:
            self.classifier = nn.Linear(self.config.hidden_size, NUM_CLASSES)

        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.lin_layer = nn.Linear(self.config.hidden_size, NUM_CLASSES)

    def forward(self, inputs):
        model_output = self.base_model(inputs)
        features = model_output.last_hidden_state  

        features = features.permute(0, 2, 1)

        pooled_features = self.pooler(features).squeeze(-1)
        outs = self.lin_layer(pooled_features)
        return outs
    

class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=True):
        super(EfficientNet_b0, self).__init__()
    
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
    

class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False):
        super(EfficientNet_b4, self).__init__()
    
        self.model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        num_ftrs = self.model.classifier[1].in_features

        if add_intermediate_layer:
            print("Adding intermediate layer...")
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
        # self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = models.resnet50(weights=None)

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