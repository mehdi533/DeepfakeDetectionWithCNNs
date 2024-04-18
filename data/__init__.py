import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import get_dataset_metadata #, dataset_folder

# def get_dataset(opt):
#     dset_lst = []
#     for cls in opt.classes:
#         root = opt.dataroot + '/' + cls
#         dset = dataset_folder(opt, root)
#         dset_lst.append(dset)
#     return torch.utils.data.ConcatDataset(dset_lst)


# -------------------------------------------------------------------------------------------

import os


def get_dataset_from_txt(opt, __type):
    images_list = get_images_list(opt, __type)
    # Returns a list of tuples. e.g. [(filename1, 0), (filename2, 0), ...]
    dataset = get_dataset_metadata(opt, images_list)
    return dataset
            
        
def get_images_list(opt, __type):

    images_list = []  # [(filename1, 0), (filename2, 0), ...]

    path = opt.metadata + __type
    
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

    return images_list

# -------------------------------------------------------------------------------------------

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt, _type):
    # shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False

    # print(shuffle)
    
    # dataset = get_dataset(opt) if not opt.metadata else get_dataset_from_txt(opt, __type)

    dataset = get_dataset_from_txt(opt, _type)
    # sampler = get_bal_sampler(dataset) if opt.class_bal else None

    # print(sampler)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                            #   shuffle=shuffle,
                                            #   sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
