import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from dataloader import create_dataloader


def metrics(y_true, y_pred, threshold=0.5):
    
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred > threshold, average='binary')
    prec = precision_score(y_true, y_pred > threshold)
    recall = recall_score(y_true, y_pred > threshold)
    auc_score = roc_auc_score(y_true, y_pred)

    return acc, ap, r_acc, f_acc, f1score, auc_score, prec, recall, y_true, y_pred

def validate(model, opt, __type):

    # Loading validation data
    data_loader = create_dataloader(opt, __type)

    with torch.no_grad():

        y_true, y_pred = [], []

        for img, label in data_loader:

            try:
                in_tens = img.cuda()
            except:
                in_tens = img
            
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    # y_true contains the label, y_pred the prediction for the label (float)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return metrics(y_true, y_pred)
