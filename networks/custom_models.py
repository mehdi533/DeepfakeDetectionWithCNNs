import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B4_Weights
from torchvision.models import ConvNeXt_Base_Weights
from transformers import AutoModel, AutoConfig
from transformers import Blip2Processor, Blip2VisionModel
from transformers import BitImageProcessor, BitForImageClassification
import timm

            
def load_custom_model(name: str, intermediate, intermediate_dim, freeze, pre_trained):

    model = None

    # Empty, fine-tune, freeze
    if name == 'res50':
        model = ResNet50(add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)
    
    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == 'vgg16':
        model = VGG16(add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)
    
    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == 'efficient_b0':
        model = EfficientNet_b0(add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == 'efficient_b4':
        model = EfficientNet_b4(add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)
    
    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == "bit":
        model = BiTModel("google/bit-50", add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)
    
    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == "vit_base":
        model = HuggingModel("google/vit-base-patch16-224", add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == "deit_base":
        model = HuggingModel("facebook/deit-base-distilled-patch16-224", add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == "coatnet":
        model = CoatNetModel('coatnet_0_rw_224.sw_in1k', add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == "resnext":
        model = ResNextModel('resnext101_32x8d', add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained) 

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == 'beit':
        model = HuggingModel("microsoft/beit-base-patch16-224", add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == 'convnext':
        model = ConvNeXt_base(add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

    # Empty, fine-tune, freeze (Comparison with other models)
    elif name == 'regnet':
        model = RegNetModel('regnet_y_400mf', add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freeze=freeze, pre_trained=pre_trained)

    # Forensics++, Voting, ... Best model so far so used multiple times and compared to all the others
    elif name == 'swin_tiny':
        model = HuggingModel("microsoft/swin-tiny-patch4-window7-224", add_intermediate_layer=intermediate, intermediate_dim=intermediate_dim, freezed=freeze, pre_trained=pre_trained)

# --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- 

    elif name == 'swin_base':
        model = HuggingModel("microsoft/swin-base-patch4-window7-224")

    # A bit too big so not used as much... Overfits a lot.
    elif name == 'swin_large':
        model = HuggingModel("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")

    elif name == "vit_large":
        model = HuggingModel("google/vit-large-patch16-224")

    elif name == "deit_small":
        model = HuggingModel("facebook/deit-small-distilled-patch16-224")

    elif name == "imagegpt_small":
        model = HuggingModel(base_mod_name='openai/imagegpt-small')
    
    elif name == "blip2":
        # Load the processor and model
        # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2VisionModel.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2VisionForImageClassification(model)

    else:
        raise ValueError("Architecture name doesn't correspond, please check utils.py for available ones.")
    return model


class Blip2VisionForImageClassification(nn.Module):
    def __init__(self, base_model, num_classes=1):
        super(Blip2VisionForImageClassification, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token
        logits = self.classifier(pooled_output)
        return logits


class RegNetModel(nn.Module):
    def __init__(self, model_name='regnet_y_400mf', num_classes=1, init_gain=0.02, pre_trained=True, freeze=False, add_intermediate_layer=True, intermediate_dim=512):
        super(RegNetModel, self).__init__()
        # Load the pre-trained RegNet model from torchvision
        self.model = getattr(models, model_name)(pretrained=pre_trained)

        # Freeze the model's parameters if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features

        if add_intermediate_layer:
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, num_classes)
            )
            torch.nn.init.normal_(self.model.fc[0].weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.fc[2].weight.data, 0.0, init_gain)
        else:
            self.model.fc = nn.Linear(in_features, num_classes)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, init_gain)
    
    def forward(self, x):
        return self.model(x)
    

class HuggingModel(nn.Module):
    def __init__(self, base_mod_name, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(base_mod_name)
        self.base_model = AutoModel.from_pretrained(base_mod_name, config=self.config)
        
        if not pre_trained:
            self.reset_parameters(init_gain)
    
        if hasattr(self.config, 'hidden_size'):
            hidden_size = self.config.hidden_size
        elif hasattr(self.base_model.classifier, 'in_features'):
            hidden_size = self.base_model.classifier.in_features
        elif hasattr(self.base_model.classifier, 'in_channels'):
            hidden_size = self.base_model.classifier.in_channels
        else:
            raise AttributeError(f"The base model's classifier does not have 'in_features' or 'in_channels' attributes.")
        
        if freezed:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        if add_intermediate_layer:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, num_classes)
            )
            torch.nn.init.normal_(self.classifier[0].weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.classifier[2].weight.data, 0.0, init_gain)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
            torch.nn.init.normal_(self.classifier.weight.data, 0.0, init_gain)
        
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.lin_layer = nn.Linear(hidden_size, num_classes)
    
    def reset_parameters(self, init_gain):
        print("Reseting parameters...")
        for module in self.base_model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight.data)
                torch.nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, inputs):
        model_output = self.base_model(inputs)
        features = model_output.last_hidden_state  

        features = features.permute(0, 2, 1)

        pooled_features = self.pooler(features).squeeze(-1)
        outs = self.lin_layer(pooled_features)
        return outs

# Empty, fine tune, freeze (with additional layer)
class ConvNeXt_base(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super(ConvNeXt_base, self).__init__()

        if pre_trained:
            # Load pre-trained model
            self.model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        else:
            # Load empty (initialized) model
            self.model = models.convnext_base(weights=None) # Random initialization

        # Freeze all layers except the last one
        if freezed:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[2].in_features

        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            self.model.classifier[2] = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dim, num_classes)  # Output layer kNN voting
            )
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.classifier[2][2].weight.data, 0.0, init_gain)
        else:
            self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.classifier[2].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)


class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes, add_intermediate_layer=False, intermediate_dim=None, init_gain=0.02):

        super(CustomHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten() 
        
        if add_intermediate_layer:
            self.fc = nn.Sequential(
                nn.Linear(in_features, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, num_classes)
            )
            torch.nn.init.normal_(self.fc[0].weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.fc[2].weight.data, 0.0, init_gain)
        else:
            self.fc = nn.Linear(in_features, num_classes)
            torch.nn.init.normal_(self.fc.weight.data, 0.0, init_gain)

    def forward(self, x, pre_logits=None):
        # Check if pre_logits is callable and use it if so
        if callable(pre_logits):
            x = pre_logits(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Empty, fine tune, freeze (with additional layer)
class CoatNetModel(nn.Module):
    def __init__(self, base_mod_name, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super().__init__()
        
        self.base_model = timm.create_model(base_mod_name, pretrained=pre_trained)
        
        if freezed:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.head = CustomHead(768, num_classes, add_intermediate_layer, intermediate_dim, init_gain)
        self.base_model.head = self.head
    
    def forward(self, inputs):
        return self.base_model(inputs)

# Empty, fine tune, freeze (with additional layer)
class ResNextModel(nn.Module):
    def __init__(self, base_mod_name, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super().__init__()
        
        self.base_model = timm.create_model(base_mod_name, pretrained=pre_trained)
        
        # if not pre_trained:
        #     self.reset_parameters(init_gain)
        
        in_features = self.base_model.fc.in_features
        
        if freezed:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        if add_intermediate_layer:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, num_classes)
            )
            torch.nn.init.normal_(self.classifier[0].weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.classifier[2].weight.data, 0.0, init_gain)
        else:
            self.classifier = nn.Linear(in_features, num_classes)
            torch.nn.init.normal_(self.classifier.weight.data, 0.0, init_gain)
        
        self.base_model.fc = self.classifier
    
    def reset_parameters(self, init_gain):
        print("Reseting parameters...")
        for module in self.base_model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
    
    def forward(self, inputs):
        return self.base_model(inputs)
    
# Empty, fine tune, freeze (with additional layer)
class BiTModel(nn.Module):
    def __init__(self, base_mod_name, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super().__init__()

        self.base_model = BitForImageClassification.from_pretrained(base_mod_name)

        if not pre_trained:
            self.reset_parameters(init_gain)
        
        if isinstance(self.base_model.classifier, nn.Sequential):
            for layer in self.base_model.classifier:
                if isinstance(layer, nn.Linear):
                    hidden_size = layer.out_features
                    break
        elif hasattr(self.base_model.classifier, 'in_features'):
            hidden_size = self.base_model.classifier.in_features
        else:
            raise ValueError("Cannot determine hidden size from the model classifier")

        # Freeze all layers except the one we will add later
        if freezed:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Option to add an additional layer (intermediate dimension)
        if add_intermediate_layer:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, num_classes)
            )
            torch.nn.init.normal_(self.classifier[0].weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.classifier[2].weight.data, 0.0, init_gain)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
            torch.nn.init.normal_(self.classifier.weight.data, 0.0, init_gain)

    def reset_parameters(self, init_gain):
        print("Reseting parameters...")
        for module in self.base_model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight.data)
                torch.nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        logits = outputs.logits
        return self.classifier(logits)
    
# Empty, fine tune, freeze (with additional layer)
class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super(EfficientNet_b0, self).__init__()

        if pre_trained:
            # Load pre-trained model
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            # Load empty (initialized) model
            self.model = models.efficientnet_b0(weights=None) # Random initialization

        # Freeze all layers except last one
        if freezed:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[1].in_features

        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            self.model.classifier[1] = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dim, num_classes)  # Output layer kNN voting
            )
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.classifier[1][2].weight.data, 0.0, init_gain)
        else:
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.classifier[1].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)
    
# Empty, fine tune, freeze (with additional layer)
class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super(EfficientNet_b4, self).__init__()

        if pre_trained:
            # Load pre-trained model
            self.model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        else:
            # Load empty (initialized) model
            self.model = models.efficientnet_b4(weights=None) # Random initialization

        # Freeze all layers except last one
        if freezed:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[1].in_features

        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            self.model.classifier[1] = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dim, num_classes)  # Output layer kNN voting
            )
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.classifier[1][2].weight.data, 0.0, init_gain)
        else:
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.classifier[1].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)

# Empty, fine tune, freeze (with additional layer)
class VGG16(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super(VGG16, self).__init__()
        
        if pre_trained:
            # Load pre-trained model
            self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            # Load empty (initialized) model
            self.model = models.vgg16(weights=None) # Random initialization

        # Freeze all layers except last one
        if freezed:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[6].in_features
        
        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            self.model.classifier[6] = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True), 
                nn.Linear(intermediate_dim, num_classes)  # Output layer kNN voting
            )
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.classifier[6][2].weight.data, 0.0, init_gain)
        else:
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.classifier[6].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)
    
# Empty, fine tune, freeze (with additional layer)
class ResNet50(nn.Module):
    def __init__(self, num_classes=1, init_gain=0.02, intermediate_dim=64, add_intermediate_layer=False, freezed=False, pre_trained=True):
        super(ResNet50, self).__init__()
        
        if pre_trained:
            # Load pre-trained model
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            # Load empty (initialized) model
            self.model = models.resnet50(weights=None) # Random initialization

        # Freeze all layers except last one
        if freezed:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        
        if add_intermediate_layer:
            self.intermediate_layer = nn.Linear(num_ftrs, intermediate_dim)
            self.model.fc = nn.Sequential(
                self.intermediate_layer,
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dim, num_classes)  # Output layer / Voting with kNN
            )
            torch.nn.init.normal_(self.intermediate_layer.weight.data, 0.0, init_gain)
            torch.nn.init.normal_(self.model.fc[2].weight.data, 0.0, init_gain)
        else:
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(x)