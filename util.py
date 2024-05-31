import os
import torch

# list of different possible model architectures
models_names = ["efficient_b0", "efficient_b4", "vgg16", "res50", "resnext", "coatnet", "swin_large", "swin_base", "swin_tiny", "vit_base", "vit_large", "deit_base", "deit_small", "bit"]

# list of different possible data
list_data = ["FFpp1", "FFpp2", "FFpp3", "FFpp4", "StyleGAN", "VQGAN", "PNDM", "DDPM", "LDM", "DDIM", "ProGAN"]

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]
