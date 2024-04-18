import functools
import torch
import torch.nn as nn
# from networks.resnet import resnet50
from networks.base_model import BaseModel #, init_weights
# --------------------------------------------------------
import torchvision.models as models
from torchvision.models import VGG16_Weights
from networks.custom_models import *
from deepfake_detector import return_model
# --------------------------------------------------------


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        # if self.isTrain and not opt.continue_train:
        self.model = return_model(opt.arch, add=opt.intermediate, dim=opt.intermediate_dim)

        # if not self.isTrain or opt.continue_train:
        #     self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            # opt.optim = 'adam'
            # if opt.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            # elif opt.optim == 'sgd':
            #     self.optimizer = torch.optim.SGD(self.model.parameters(),
            #                                      lr=opt.lr, momentum=0.0, weight_decay=0)
            # else:
            #     raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        try:
            self.model.to(self.device)
        except IndexError:
            self.model.to('cpu')


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    # def get_loss(self):
    #     return self.loss_fn(self.output.squeeze(1), self.label)
        # return self.loss_fn(self.output, self.label)

    def optimize_parameters(self):
        self.forward()
        # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

