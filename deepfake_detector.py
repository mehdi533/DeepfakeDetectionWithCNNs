import os
import numpy as np
import torch
from networks.custom_models import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import timm
import torch.nn as nn

def return_model(model, add=False, dim=64):
    if model == "res50":
        return ResNet50(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "vgg16":
        return VGG16(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "efficient_b0":
        return EfficientNet_b0(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "efficient_b4":
        return EfficientNet_b4(add_intermediate_layer=True, intermediate_dim=dim)

    elif model == 'swin_tiny':
        return HuggingModel("microsoft/swin-tiny-patch4-window7-224", 1) #["base_model.encoder.layers.3.blocks.1"]

    elif model == 'swin_base':
        return HuggingModel("microsoft/swin-base-patch4-window7-224", 1)

    elif model == 'swin_large':
        return HuggingModel("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft", 1)

    elif model == "coatnet":
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
        # Assuming you're using the model 'coatnet_0_rw_224.sw_in1k' and want NUM_CLASSES as output
        model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True)
        NUM_CLASSES = 1  # Set the number of classes as per your dataset requirement
        # Replace the 'head' with your new custom head
        model.head = CustomHead(in_features=768, num_classes=NUM_CLASSES)
        return model
    elif model == "resnext":
        net = timm.create_model('resnext101_32x8d', pretrained=True)
        # Assuming the last layer was named 'head' and you verified it has an attribute 'in_features'
        net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, net.fc.in_features),
            nn.ReLU(),
            nn.Linear(net.fc.in_features, 1)
        )
        return net
    else:
        raise ValueError("Model name should either be res50, vgg16, efficient_b0, efficient_b4, or swin")


def load_model(path):
    truncs = path.split('_')
    model, _type, trained_on = truncs[0], truncs[1], truncs[2]

    if _type == "inter#64":
        add = True
    else:
        add = False
    
    return return_model(model, _type)


def voting_prediction(list_of_predictions):

    list_of_predictions = list_of_predictions.T

    image_score = []

    for i in range(list_of_predictions.shape[0]):
        score = max(list_of_predictions[i, :])
        image_score.append(score)

    return np.array(image_score)


class Detector():

    def __init__(self, list_path_models: str):

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        
        for i, model_path in enumerate(list_path_models):

            print(f'Loading model {model_path.split("/")[-2]}...')

            net = load_model(model_path.split('/')[-2])
            state_dict = torch.load(model_path, map_location='cpu')
            net.load_state_dict(state_dict['model'])

            net.cuda().eval()

            self.nets += [net]

    def synth_real_detector(self, data_loader):
        
        y_true, y_pred_array = [], []

        with torch.no_grad():    
            
            for net_idx, net in enumerate(self.nets):
                y_pred_temp = []

                for img, label in data_loader:

                    if net_idx == 0:
                        y_true.extend(label.flatten().tolist())

                    in_tens = img.cuda()
                    y_pred_temp.extend(np.array(net(in_tens).sigmoid().flatten().tolist()))

                y_pred_array.append(np.array(y_pred_temp))

        y_true = np.array(y_true)
        y_pred = voting_prediction(np.array(y_pred_array))

            # for img, label in data_loader:
                
            #     y_true.extend(label.flatten().tolist())
            #     img_net_scores = []

            #     for net_idx, net in enumerate(self.nets):
            #         print(net_idx)
            #         in_tens = img.cuda()
            #         pred_temp = np.array(net(in_tens).sigmoid().flatten().tolist())
            #         pred_temp_bin = pred_temp > 0.5

            #         maj_voting = np.any(pred_temp_bin).astype(int)
            #         scores_maj_voting = pred_temp[maj_voting]
            #         img_net_scores.extend(np.nanmax(scores_maj_voting) if maj_voting == 1 else -np.nanmax(scores_maj_voting))
                
            #     print(img_net_scores)
            #     y_pred.extend(np.mean(img_net_scores))

        return y_true, y_pred
    
    def return_metrics(self, y_true, y_pred):
        
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred > 0.5, average='binary')
        prec = precision_score(y_true, y_pred > 0.5)
        recall = recall_score(y_true, y_pred > 0.5)
        auc_score = roc_auc_score(y_true, y_pred)

        return acc, ap, r_acc, f_acc, f1score, auc_score, prec, recall, y_true, y_pred
    