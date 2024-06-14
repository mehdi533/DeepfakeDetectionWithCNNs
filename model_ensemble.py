import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from networks.custom_models import load_custom_model
from util import models_names
from dataloader import create_dataloader


class MetaModel:
    def __init__(self, name, features, y):

        self.name = name

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(features)

        if name == "kNN":
            self.meta_model = KNeighborsClassifier(n_neighbors=1, metric='manhattan', weights='distance').fit(X_train, y)
        elif name == "LR":
            self.meta_model = LinearRegression().fit(X_train, y)
        else:
            raise AttributeError('The meta model name does not correspond to the available ones: "kNN" and "LR"')

    def predict_probabilities(self, features):
        X_test = self.scaler.transform(features)
        if self.name == "kNN":
            return self.meta_model.predict_proba(X_test)[:, 1]
        elif self.name == "LR":
            return self.meta_model.predict(X_test)


def simple_averaging(predictions_array):
    combined_preds = np.mean(predictions_array, axis=0)
    return combined_preds


def weighted_averaging(predictions_array, weights):
    assert len(weights) == predictions_array.shape[0] - 1
    weighted_preds = np.average(predictions_array, axis=0, weights=weights)
    return weighted_preds


class Ensemble():

    def __init__(self, list_path_models: str, opt):
        
        # Default GPU (izar) or CPU (helvetios)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []

        print(f"The model ensemble uses: {opt.meta_model}")

        for model_path in os.listdir(list_path_models):
            
            print(f'Loading model {model_path}...')

            # Allows to use different architectures
            for possible_arch in models_names:
                if possible_arch in model_path:
                    arch = possible_arch
                    break
            
            # Load best model of all the folders available
            net = load_custom_model(arch, opt.intermediate, opt.intermediate_dim, opt.freeze, opt.pre_trained)
            state_dict = torch.load(os.path.join(list_path_models, model_path, "model_epoch_best.pth"), map_location='cpu')
            net.load_state_dict(state_dict['model'])

            try: # If available, send model to GPU (izar cluster), otherwise use CPU (helvetios)
                net.cuda().eval()
            except:
                net.eval()

            self.nets += [net]
            
        if opt.meta_model == "kNN" or opt.meta_model == "LR":
            
            # Training the meta model on the validation set
            data_loader_val = create_dataloader(opt, "val_list")
  
            X_train, y_train = self.extract_features_from_models(data_loader_val)
            
            print("Training meta model...")
            self.meta_model = MetaModel(opt.meta_model, X_train, y_train)
        
        else:
            self.meta_model = "average"

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

                    # extract features function not defined for all models (hugging face models and convnext)
                    features = net.extract_features(in_tens).cpu().detach().numpy()
                    net_features.extend(features)
                
                features_list.append(np.array(net_features))

        combined_features = np.concatenate(features_list, axis=1)

        y_true = np.array(y_true)
        return combined_features, y_true

    def synth_real_detector(self, data_loader):
        
        y_true, y_pred_array = [], []

        if self.meta_model == "average":

            with torch.no_grad():    
                for net_idx, net in enumerate(self.nets):
                    y_pred_temp = []

                    for img, label in data_loader:
                        
                        if net_idx == 0: # Only need y_true once
                            y_true.extend(label.flatten().tolist())

                        try:
                            in_tens = img.cuda()
                        except:
                            in_tens = img
                    
                        y_pred_temp.extend(np.array(net(in_tens).sigmoid().flatten().tolist()))

                    # Adds the prediction of the model to the array of predictions
                    # Format: [[preds_model1], [preds_model2], ...]
                    y_pred_array.append(np.array(y_pred_temp))

            y_true = np.array(y_true)

            y_pred = simple_averaging(y_pred_array)

        else:

            X_test, y_true = self.extract_features_from_models(data_loader)

            # Predict probabilities on the test set
            y_pred = self.meta_model.predict_probabilities(X_test)
            
        return y_true, y_pred
    