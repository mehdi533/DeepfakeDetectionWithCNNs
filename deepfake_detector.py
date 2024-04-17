import os
import numpy as np
import torch
from networks.custom_models import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def return_model(model, add=False, dim=64):
    if model == "res50":
        return ResNet50(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "vgg16":
        return VGG16(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "efficient_b0":
        return EfficientNet_b0(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "efficient_b4":
        return EfficientNet_b4(add_intermediate_layer=add, intermediate_dim=dim)
    elif model == "swin":
        return HuggingModel("microsoft/swin-tiny-patch4-window7-224", additional_layers=add)
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
    