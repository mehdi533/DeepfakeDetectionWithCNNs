import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
#-----------–-----------–-----------–--------
import torchvision.models as models
import torch.nn as nn
from networks.custom_models import *
from deepfake_detector import Detector
from data import create_dataloader
#-----------–-----------–-----------–--------
from options.test_options import TestOptions
from eval_config import *


def evaluation(model_path, name, opt):
    # Running tests
    model_name = os.path.basename(model_path).replace('.pth', '')
    rows = [["{} model testing on...".format(name)],
            ['testset', 'accuracy', 'avg precision', "f1 score", "roc score", "recall", "precision"]]

    print("{} model testing on...".format(name))

    # ---------------------------------------------------------------------
    opt.models = ["real", "PNDM", "DDPM", "LDM", "ProGAN"]
    list_models = opt.models
    list_models.remove("real")

    for v_id, test_model in enumerate(list_models):
            
        #opt.dataroot = '{}/{}'.format(dataroot, val)
        #opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        #opt.classes = [0, 1]
    # ---------------------------------------------------------------------
        opt.no_resize = True    # testing without resizing by default

        # model = resnet50(num_classes=1)
        # model = models.efficientnet_b0(num_classes=1)
        # num_ftrs = model.classifier[1].in_features
        # model.classifier[1] = nn.Linear(num_ftrs, 1)
        # model.fc = nn.Linear(2048, 1)
        # model = models.vgg16(num_classes=1)
        model = VGG16(num_classes=1)
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model.cuda()
        model.eval()
        opt.models = [test_model, "real"]
        # ---------------------------------------------------------------------
        acc, ap, r_acc, f_acc, f1, auc, prec, recall, _, _ = validate(model, opt, "test_list")
        rows.append([test_model, acc, ap, f1, auc, prec, recall])
        print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {} f1: {}; roc_auc: {}; recall: {}; precision: {}".format(test_model, acc, ap, r_acc, f_acc, f1, auc, recall, prec))
        # ---------------------------------------------------------------------

    csv_name = results_dir + '/{}.csv'.format(name)
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)


def eval_multiple():

    rows = [["{} model testing on...".format(opt.name)],
            ['testset', 'accuracy', 'avg precision', "f1 score", "roc score", "recall", "precision"]]

    print("{} model testing on...".format(opt.name))

    # ---------------------------------------------------------------------
    opt.models = ["real", "PNDM", "DDPM", "LDM", "ProGAN"]
    list_models = opt.models
    list_models.remove("real")

    detector = Detector()
    data_loader = create_dataloader(opt, "test_list")
    y_true, y_pred = detector.synth_real_detector(data_loader)
    acc, ap, r_acc, f_acc, f1, auc, prec, recall, _, _ = detector.return_metrics(y_true, y_pred)

    rows.append([test_model, acc, ap, f1, auc, prec, recall])
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {} f1: {}; roc_auc: {}; recall: {}; precision: {}".format(test_model, acc, ap, r_acc, f_acc, f1, auc, recall, prec))



if __name__ == "__main__":
    opt = TestOptions().parse(print_options=False)
    evaluation(opt.model_path, opt.name, opt)
