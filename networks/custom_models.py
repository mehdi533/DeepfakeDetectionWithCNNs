import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B4_Weights
from torchvision.models.resnet import ResNet50_Weights
from transformers import SwinForImageClassification
from transformers import AutoModel , AutoConfig, AutoTokenizer

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, freeze_layers=True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(SwinTransformer, self).__init__()
        model_name = 'microsoft/swin-tiny-patch4-window7-224'
        self.model = SwinForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

        if freeze_layers:
            # Freeze all layers initially
            for name, param in self.model.named_parameters():
                # Freeze parameters by default
                param.requires_grad = False

                # Unfreeze parameters in the last stage (stage 3, in this case)
                if 'encoder.layers.3' in name:
                    param.requires_grad = True

    def forward(self, x, *args, **kwargs):
        outputs = super().__call__(x,  *args, **kwargs)
        # Directly return the logits or any other specific output component
        return outputs.logits  # Or return outputs if you want the full output object
    

class HuggingModel(nn.Module):
    def __init__(self, base_mod_name, NUM_CLASSES, freeze_layers=None, additional_layers=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_mod_name)
        self.base_model = AutoModel.from_pretrained(base_mod_name, config=self.config)
        
        # Freeze layers if specified
        if freeze_layers:
            for name, param in self.base_model.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
        
        if additional_layers:
            # Add one or two linear layers for classification
            self.classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, NUM_CLASSES)
            )
        else:
            # Directly connect to a single linear layer for classification
            self.classifier = nn.Linear(self.config.hidden_size, NUM_CLASSES)

        # Assuming the last hidden state has a shape of [batch, sequence_length, hidden_size]
        # and you want to pool this to a shape of [batch, hidden_size]
        self.pooler = torch.nn.AdaptiveAvgPool1d(1)
        self.lin_layer = torch.nn.Linear(self.config.hidden_size, NUM_CLASSES)

    def forward(self, inputs):
        model_output = self.base_model(inputs)
        features = model_output.last_hidden_state  # Assuming shape [batch, seq_len, hidden_size]
        # Convert features from [batch, seq_len, hidden_size] to [batch, hidden_size, seq_len]
        features = features.permute(0, 2, 1)
        # Pool the features to shape [batch, hidden_size, 1] and remove the last dimension
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