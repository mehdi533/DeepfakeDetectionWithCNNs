import torch
import torch.nn as nn

from networks.base_model import BaseModel
from networks.custom_models import *


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain:
            # Calls function to load the model with corresponding architecture
            self.model = load_custom_model(opt.arch, opt.intermediate, opt.intermediate_dim)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            
            # Pre-defined betas and learning rate
            betas = (0.9, 0.999)
            lr = 0.0001

            # Optimizer set as Adam
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,  betas=betas)

        # For validation loads the latest model
        if not self.isTrain:
            self.load_networks("latest")

        # Send model to device
        self.model.to(self.device)

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

