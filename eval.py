import os
import csv
import torch

from util import *
from validate import validate, metrics
from networks.custom_models import load_custom_model
from model_voting import Voting
from data import create_dataloader
from options import Options

results_dir = './results/'
mkdir(results_dir)


def evaluation(model_path, exp_name, opt):
    
    for possible_arch in models_names:
        if possible_arch in model_path:
            arch = possible_arch
            break
    
    list_present = []
    for present_data in list_data:
        if present_data in model_path:
            list_present.append(present_data)
            
    name = exp_name + '_' + arch + "-" + '-'.join(list_present)

    rows = [["{} model testing on...".format(model_path.split("/")[-2])],
            ['testset', 'accuracy', 'avg precision', "f1 score", "roc score", "precision", "recall"]]

    print("{} model testing on...".format(name))
    
    model = load_custom_model(arch, opt.intermediate, opt.intermediate_dim, opt.freeze, opt.pre_trained)
        
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    
    try: # If available, send model to GPU (izar cluster), otherwise use CPU (helvetios)
        model.cuda()
        model.eval()
    except:
        model.eval()
    
    for test_model in list_data: # list_data is in utils.py
            
        opt.no_resize = True    

        # If testing on Forensics++ dataset, use real images from Forensics++, otherwise from CelebAMask-HQ
        if "FFpp" in test_model:
            opt.models = [test_model, "FFpp0"]
        else: 
            opt.models = [test_model, "real"]
        
        # returns: acc, ap, r_acc, f_acc, f1score, auc_score, prec, recall, y_true, y_pred
        acc, ap, r_acc, f_acc, f1, auc, prec, recall, _, _ = validate(model, opt, "test_list")

        # Apprend rows to save as csv
        rows.append([test_model, acc, ap, f1, auc, prec, recall])
        print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {} f1: {}; roc_auc: {}; recall: {}; precision: {}".format(test_model, acc, ap, r_acc, f_acc, f1, auc, recall, prec))

    csv_name = results_dir + '/{}.csv'.format(name)
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)


# Double Check This

def eval_voting(path_list, exp_name, opt):

    arch_models = []

    # Formatize the name of the file to save
    # ExperimentName_ARCH1-TRAININGMODELS_ARCH2-TRAININGMODELS...
    for models in os.listdir(path_list):
        arch = models.split('_')[0]
        training_data = models.split('_')[-1]
        arch_models.append((arch, training_data))

    name = exp_name + '_' + '_'.join([val[0] + "-" + val[1] for val in arch_models])

    rows = [["{} testing on...".format(name)],
            ['testset', 'accuracy', 'avg precision', "f1 score", "roc score", "recall", "precision"]]

    print("{} testing on...".format(name))

    detector = Voting(path_list, opt)

    for test_model in list_data:

        if "FFpp" in test_model:
            opt.models = [test_model, "FFpp0"]
        else: 
            opt.models = [test_model, "real"]

        data_loader = create_dataloader(opt, "test_list")

        y_true, y_pred = detector.synth_real_detector(data_loader)
        acc, ap, r_acc, f_acc, f1, auc, prec, recall, _, _ = metrics(y_true, y_pred)

        rows.append([test_model, acc, ap, f1, auc, prec, recall])
        print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {} f1: {}; roc_auc: {}; recall: {}; precision: {}".format(test_model, acc, ap, r_acc, f_acc, f1, auc, recall, prec))

    csv_name = results_dir + '/{}.csv'.format(name)
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)


if __name__ == "__main__":
    opt = Options().parse(print_options=False)
    
    # If given a folder, will use voting function to determine result
    if "pth" in opt.path:
        evaluation(opt.path, opt.name, opt)
    else:
        eval_voting(opt.path, opt.name, opt)
