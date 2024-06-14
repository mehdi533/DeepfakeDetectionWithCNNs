import os
import torch
from dataloader.datasets import get_dataset_metadata
from dataloader.sampling import *


def get_dataset_from_txt(opt, mode):
    
    images_list = get_images_list(opt, mode)

    # Returns a list of tuples. e.g. [(filename1, 0), (filename2, 0), ...]
    dataset = get_dataset_metadata(opt, images_list)

    return dataset


def get_images_list(opt, mode):

    images_list = []  # [(filename1, 0), (filename2, 0), ...]

    path = opt.metadata + mode
    
    for cls in os.listdir(path):
        root_metadata = os.path.join(path, cls)
        # e.g. metadata/real/

        for filename in os.listdir(root_metadata):
            # e.g. metadata/train/DDPM/
            file_path = os.path.join(root_metadata, filename)
            
            model = filename.split("_")[0]

            with open(file_path) as f:
                for line in f.readlines():
                    if model in opt.models:
                        images_list.append((cls.split("_")[0], os.path.join(opt.dataroot, model, line.strip())))
                        # e.g. (0, dataset/DDIM/00000.jpg)

    if mode == "train_list":
        return multiply_class(images_list, '0', multiplier=opt.multiply_real)
    else:
        return images_list
    

def create_dataloader(opt, mode):

    dataset = get_dataset_from_txt(opt, mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_threads)
    
    return data_loader
