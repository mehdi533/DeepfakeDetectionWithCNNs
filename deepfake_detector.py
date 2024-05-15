import numpy as np
import torch
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from networks.custom_models import load_custom_model


def voting_prediction(list_of_predictions):
    """
    In this function is defined the how each model influences the 
    attributed class of the image.
    """
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

            net = load_custom_model(model_path.split('/')[-2])
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
    