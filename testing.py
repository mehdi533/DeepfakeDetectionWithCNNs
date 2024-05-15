"""
This script is used to test mutliple models at the same time, all in the same dir
"""

import os
from eval import evaluation
from options.test_options import TestOptions
from util import *

def model_arch(dir):
    for name in models_names:
        if name in dir:
            return name

if __name__ == "__main__":

    opt = TestOptions().parse(print_options=False)
    # eval_resnet50-DDPM-PNDM_bs256
    # /home/abdallah/code/checkpoints/resnet50/
    # resnet50_bs256_DDPM-PNDM/model_epoch_best.pth
    # python testing.py --name eval_resnet50 --batch_size 256 --model_path /home/abdallah/code/checkpoints/resnet50
    
    for dir in os.listdir(opt.folder_path):
        
        """ 
        resnet50_name_GAN1-GAN2 
        efficient_b0_name_GAN1-GAN2
        """
        
        model_path = os.path.join(opt.folder_path, dir, "model_epoch_best.pth")

        # Possible models:
        # res50, vgg16, resnext, swin_tiny, swin_base, swin_large, coatnet, efficient_b0, efficient_b4
        
        opt.arch = model_arch(dir)

        evaluation(model_path, dir, opt)
