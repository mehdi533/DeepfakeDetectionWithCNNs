import torch
import numpy as np
# from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from options.test_options import TestOptions
from data import create_dataloader
from deepfake_detector import return_model


def validate(model, opt, __type):
    # --------------------------------------------------------------------------------
    data_loader = create_dataloader(opt, __type)
    # --------------------------------------------------------------------------------
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            # --------------------------------------------------------------------------------
            try:
                in_tens = img.cuda()
            except AssertionError:
                in_tens = img
            # --------------------------------------------------------------------------------
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ---------------------------------------- ----------------------------------------
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred > 0.5, average='binary')
    prec = precision_score(y_true, y_pred > 0.5)
    recall = recall_score(y_true, y_pred > 0.5)
    auc_score = roc_auc_score(y_true, y_pred)
    # ---------------------------------------- ----------------------------------------
    return acc, ap, r_acc, f_acc, f1score, auc_score, prec, recall, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = return_model(opt.arch, add=opt.intermediate, dim=opt.intermediate_dim)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    try:
        model.cuda()
    finally:
        model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
