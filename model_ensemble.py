import numpy as np
import os
import torch

from networks.custom_models import load_custom_model
from util import models_names
from dataloader import create_dataloader

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, make_scorer

class IntermediateModel(nn.Module):
    def __init__(self, original_model):
        super(IntermediateModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    
    def forward(self, x):
        return self.features(x)

def feature_space(model, data):
    intermediate_model = IntermediateModel(model)
    features = intermediate_model(data)
    print(features.shape)

def train_meta_model(base_model_predictions, y_val):
    # meta_features = np.vstack(base_model_predictions).T
    meta_model = LinearRegression().fit(base_model_predictions, y_val)
    # Initialize KNN
    # meta_model = KNeighborsClassifier(n_neighbors=10).fit(base_model_predictions, y_val)

    # meta_features_df = pd.DataFrame(meta_features, columns=['Model1_Pred', 'Model2_Pred', 'Model3_Pred'])
    # meta_features_df['True_Label'] = y_val
    # meta_features_df.to_csv('results/meta_features.csv', index=False)
    # Coefficients (slope)
    # print("Coefficients:", meta_model.coef_)

    # # Intercept
    # print("Intercept:", meta_model.intercept_)

    return meta_model


def predict_with_meta_model(base_model_predictions, meta_model):
    # meta_features = np.vstack(base_model_predictions).T
    meta_predictions = meta_model.predict_proba(base_model_predictions)[:,1]
    print(meta_predictions)
    return meta_predictions


def weighted_averaging(predictions_array, weights):
    assert len(weights) == predictions_array.shape[0] - 1
    weighted_preds = np.average(predictions_array, axis=0, weights=weights)
    return weighted_preds


def simple_averaging(predictions_array):
    combined_preds = np.mean(predictions_array, axis=0)
    return combined_preds


def transform_values(data, threshold=0.3, lower_factor=0.5, upper_factor=0.5):
    """
    Transforms the input data such that values close to the threshold are pushed down
    closer to 0 and values above the threshold are pushed closer to 1.

    Parameters:
    - data: array-like, the input data to be transformed.
    - threshold: float, the threshold value to separate the data.
    - lower_factor: float, the factor by which to reduce values below the threshold.
    - upper_factor: float, the factor by which to increase values above the threshold.

    Returns:
    - transformed_data: array-like, the transformed data.
    """
    transformed_preds = []

    for preds_model in data:

        transformed_data = []

        for value in preds_model:
            if value < threshold:
                # Push values closer to 0
                new_value = value * lower_factor
            else:
                # Push values closer to 1
                new_value = 1 - (1 - value) * upper_factor
            
            transformed_data.append(new_value)
        
        transformed_preds.append(np.array(transformed_data))
    
    return np.array(transformed_preds)


def voting_prediction(list_of_predictions):

    list_of_predictions = list_of_predictions.T

    image_score = []

    for i in range(list_of_predictions.shape[0]):
        score = max(list_of_predictions[i, :])
        image_score.append(score)

    return np.array(image_score)


class Ensemble():

    def __init__(self, list_path_models: str, opt):
        
        # Default GPU (izar) or CPU (helvetios)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        
        for model_path in os.listdir(list_path_models):
            
            print(f'Loading model {model_path}...')

            for possible_arch in models_names:
                if possible_arch in model_path:
                    arch = possible_arch
                    break

            net = load_custom_model(arch, opt.intermediate, opt.intermediate_dim, opt.freeze, opt.pre_trained)
            state_dict = torch.load(os.path.join(list_path_models, model_path, "model_epoch_best.pth"), map_location='cpu')
            net.load_state_dict(state_dict['model'])

            try: # If available, send model to GPU (izar cluster), otherwise use CPU (helvetios)
                net.cuda().eval()
            except:
                net.eval()

            self.nets += [net]
            
            # net = load_custom_model('swin_tiny', 1, 64, opt.freeze, opt.pre_trained)
            # state_dict = torch.load("/home/abdallah/Deepfake-Detection/models_trained/swin_tiny_freeze_0606_ProGAN/model_epoch_best.pth", map_location='cpu')
            # net.load_state_dict(state_dict['model'])
            # net.eval()
            # self.nets = [net]
        
        if opt.meta_model:
            
            # data_loader_val = create_dataloader(opt, "val_list")

            # y_true, y_pred_array, features_list = [], [], []

            # with torch.no_grad():    
                
            #     for net_idx, net in enumerate(self.nets):
            #         y_pred_temp = []
            #         net_features = []
            #         for img, label in data_loader_val:
                        
            #             # features = net.extract_features(img).sigmoid().flatten().tolist()
            #             # print(len(features))
            #             # print("passed")
            #             # Only need y_true once
            #             if net_idx == 0:
            #                 y_true.extend(label.flatten().tolist())

            #             try:
            #                 in_tens = img.cuda()
            #             except:
            #                 in_tens = img
                        
            #             # Is it better to use sigmoid or softmax?
            #             features = net.extract_features(in_tens).cpu().detach().numpy()
            #             net_features.extend(features)
            #             # y_pred_temp.extend(np.array(net(in_tens).sigmoid().flatten().tolist()))

            #         # Adds the prediction of the model to the array of predictions
            #         # Format: [[preds_model1], [preds_model2], ...]
            #         # y_pred_array.append(np.array(y_pred_temp))
            #         features_list.append(np.array(net_features))
            # y_pred_array = np.concatenate(features_list, axis=1)  # Shape: (num_samples, total_feature_dim)
            # print(y_pred_array.shape)
            # y_true = np.array(y_true)  # Shape: (num_samples,)          

            # print("Training meta model...")
            # self.meta_model = train_meta_model(y_pred_array, y_true)
            data_loader_val = create_dataloader(opt, "val_list")
            X_train, y_train = self.extract_features_from_models(data_loader_val)
            print("Features to train kNN are extracted...")
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 20, 30],
                'metric': ['euclidean', 'manhattan']
            }

            # Define the kNN model
            self.best_knn = KNeighborsClassifier(n_neighbors=10, metric='cosine', weights='distance').fit(X_train, y_train)

            # Define the scoring metric
            # scorer = make_scorer(average_precision_score, needs_proba=True)

            # Perform grid search with cross-validation
            # grid_search = GridSearchCV(self.knn, param_grid, scoring=scorer, cv=5)
            # grid_search.fit(X_train, y_train)

            # Get the best model
            # self.best_knn = grid_search.best_estimator_
            # print(f'Best parameters: {grid_search.best_params_}')
            self.meta_model = True

        else:
            self.meta_model = False

    def extract_features_from_models(self, dataloader):
        features_list = []
        y_true = []

        with torch.no_grad():
            for net_idx, net in enumerate(self.nets):
                print(f"Model number: {net_idx}")
                net_features = []
                for img, label in dataloader:
                    if net_idx == 0:
                        y_true.extend(label.flatten().tolist())

                    try:
                        in_tens = img.cuda()
                    except:
                        in_tens = img

                    features = net.extract_features(in_tens).cpu().detach().numpy()
                    net_features.extend(features)
                
                features_list.append(np.array(net_features))

        combined_features = np.concatenate(features_list, axis=1)
        y_true = np.array(y_true)
        return combined_features, y_true

    def synth_real_detector(self, data_loader):
        
        y_true, y_pred_array = [], []

        # with torch.no_grad():    
        #     for net_idx, net in enumerate(self.nets):
        #         y_pred_temp = []
        #         print("Predicting...")
        #         for img, label in data_loader:
                    
        #             # Only need y_true once
        #             if net_idx == 0:
        #                 y_true.extend(label.flatten().tolist())

        #             try:
        #                 in_tens = img.cuda()
        #             except:
        #                 in_tens = img
                    
        #             # Is it better to use sigmoid or softmax?
        #             # y_pred_temp.extend(net.extract_features(in_tens).cpu().detach().numpy())
        #             y_pred_temp.extend(np.array(net(in_tens).sigmoid().flatten().tolist()))

        #         # Adds the prediction of the model to the array of predictions
        #         # Format: [[preds_model1], [preds_model2], ...]
        #         y_pred_array.append(np.array(y_pred_temp))

        # y_true = np.array(y_true)

        if self.meta_model:
            X_test, y_true = self.extract_features_from_models(data_loader)

            # Scale the test features
            X_test = self.scaler.transform(X_test)

            # Predict probabilities on the test set
            y_pred = self.best_knn.predict_proba(X_test)[:, 1]
            print(y_pred)
            # Why no other value than 0 or 1

            # y_pred_array = np.concatenate(y_pred_array, axis=1)
            # y_pred = predict_with_meta_model(y_pred_array, self.meta_model)
        else:
            print("Averaging...")
            # print(y_pred_array)
            # y_pred_temp = transform_values(y_pred_array)
            # print(y_pred_temp)
            y_pred = simple_averaging(y_pred_array)
            y_pred = y_pred * 3
            # print(y_pred)
            y_pred[y_pred > 1] = 1


        return y_true, y_pred
    