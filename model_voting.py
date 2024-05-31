import numpy as np
import os
import torch

from networks.custom_models import load_custom_model
from util import models_names


def voting_prediction(list_of_predictions):
    """
    In this function is defined the how each model influences the attributed class of the image.
    Related work: 
    """

    # Transposing the array
    list_of_predictions = list_of_predictions.T

    image_score = []

    # Double check this (Mandelli)
    for i in range(list_of_predictions.shape[0]):
        score = max(list_of_predictions[i, :])
        image_score.append(score)

    return np.array(image_score)


class Voting():

    def __init__(self, list_path_models: str):
        
        # Default GPU (izar) or CPU (helvetios)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        
        for model_path in os.listdir(list_path_models):
            
            print(f'Loading model {model_path}...')

            for possible_arch in models_names:
                if possible_arch in model_path:
                    arch = possible_arch
                    break

            net = load_custom_model(arch, intermediate=False, intermediate_dim=0)
            state_dict = torch.load(os.path.join(list_path_models, model_path, "model_epoch_best.pth"), map_location='cpu')
            net.load_state_dict(state_dict['model'])

            try: # If available, send model to GPU (izar cluster), otherwise use CPU (helvetios)
                net.cuda().eval()
            except:
                net.eval()

            self.nets += [net]

    def synth_real_detector(self, data_loader):
        
        y_true, y_pred_array = [], []

        with torch.no_grad():    
            
            for net_idx, net in enumerate(self.nets):
                y_pred_temp = []

                for img, label in data_loader:
                    
                    # Only need y_true once
                    if net_idx == 0:
                        y_true.extend(label.flatten().tolist())

                    try:
                        in_tens = img.cuda()
                    except:
                        in_tens = img
                        
                    y_pred_temp.extend(np.array(net(in_tens).sigmoid().flatten().tolist()))

                # Adds the prediction of the model to the array of predictions
                y_pred_array.append(np.array(y_pred_temp))

        y_true = np.array(y_true)
        y_pred = voting_prediction(np.array(y_pred_array))

        return y_true, y_pred
    