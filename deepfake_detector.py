import os
import numpy as np
import torch
from networks.custom_models import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

class Detector():

    def __init__(self, path_list_models: str):

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        
        for i, model in enumerate(os.listdir(path_list_models)):
            model_path = os.path.join(path_list_models, model)
            print(f'Loading model {model}...')

            net = VGG16(num_classes=1)
            state_dict = torch.load(model_path, map_location='cpu')
            net.load_state_dict(state_dict['model'])

            net.cuda().eval()

            print('model loaded!\n')

            self.nets += [net]

    def synth_real_detector(self, data_loader):
        
        y_true, y_pred = [], []

        with torch.no_grad():    

            for img, label in data_loader:
                
                y_true.extend(label.flatten().tolist())
                img_net_scores = []

                for net_idx, net in enumerate(self.nets):
                    in_tens = img.cuda()

                    pred_temp = (net(in_tens).sigmoid().flatten().tolist())
                    pred_temp_bin = pred_temp > 0.5

                    maj_voting = np.any(pred_temp_bin).astype(int)
                    scores_maj_voting = pred_temp[:, maj_voting]
                    img_net_scores.append(np.nanmax(scores_maj_voting) if maj_voting == 1 else -np.nanmax(scores_maj_voting))

                y_pred.extend(np.mean(img_net_scores))

        y_true, y_pred = np.array(y_true), np.array(y_pred)

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
    