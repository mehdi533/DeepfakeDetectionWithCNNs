import numpy as np
import os
import torch

from networks.custom_models import load_custom_model
from util import models_names

from sklearn.linear_model import LinearRegression


def train_meta_model(base_model_predictions, true_labels):
    """
    Train a linear regression model to combine base model predictions.
    
    Args:
    base_model_predictions (np.array): An array of shape (num_samples, num_models) containing the predictions of each model.
    true_labels (np.array): The true labels for the samples.
    
    Returns:
    LinearRegression: The trained linear regression model.
    """
    # The input X is already in the required shape (num_samples, num_models)
    X = base_model_predictions
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, true_labels)

    coefficients = model.coef_
    
    return model, coefficients
    

def linear_regression_prediction(base_model_predictions, true_labels):
    """
    Combine predictions using a trained linear regression model.
    
    Args:
    base_model_predictions (np.array): An array of shape (num_samples, num_models) containing the predictions of each model.
    true_labels (np.array): The true labels for the samples.
    
    Returns:
    np.array: An array containing the final class probabilities.
    """
    # Train the meta-model
    meta_model, coefficients = train_meta_model(base_model_predictions, true_labels)

    # The input X is already in the required shape (num_samples, num_models)
    X = base_model_predictions
    
    # Predict using the meta-model
    combined_predictions = meta_model.predict(X)

    print("Coefficients for each model's prediction:", coefficients)
    
    return combined_predictions

def weighted_averaging(predictions_array, weights):
    # Ensure the weights array is the same length as the number of prediction rows minus the true labels
    assert len(weights) == predictions_array.shape[0] - 1
    # Exclude the first row which contains the true labels
    weighted_preds = np.average(predictions_array[1:], axis=0, weights=weights)
    return weighted_preds

def simple_averaging(predictions_array):
    # Exclude the first row which contains the true labels
    combined_preds = np.mean(predictions_array[1:], axis=0)
    return combined_preds


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
                    
                    # Is it better to use sigmoid or softmax?
                    y_pred_temp.extend(np.array(net(in_tens).sigmoid().flatten().tolist()))

                # Adds the prediction of the model to the array of predictions
                # Format: [[preds_model1], [preds_model2], ...]
                y_pred_array.append(np.array(y_pred_temp))

        y_true = np.array(y_true)
        y_pred = linear_regression_prediction(np.array(y_pred_array).T, y_true)

        return y_true, y_pred
    