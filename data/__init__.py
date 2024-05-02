import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder, get_dataset_metadata

def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


# -------------------------------------------------------------------------------------------

import os
from collections import defaultdict
import random

def get_dataset_from_txt(opt, __type):
    images_list = get_images_list(opt, __type)
    # Returns a list of tuples. e.g. [(filename1, 0), (filename2, 0), ...]
    dataset = get_dataset_metadata(opt, images_list)
    return dataset


def balancing(data_list):

    class_counts = defaultdict(int)

    for label, _ in data_list:
        class_counts[label] += 1
    
    min_samples = min(class_counts.values())

    # print(class_counts)

    balanced_data_list = []

    for label in class_counts:
        # Extract all samples of a class
        class_samples = [item for item in data_list if item[0] == label]
        # Shuffle to add randomness
        random.shuffle(class_samples)
        # Add equal number of samples from each class
        balanced_data_list.extend(class_samples[:min_samples])

    # Shuffle the final list to mix classes
    random.shuffle(balanced_data_list)

    class_counts = defaultdict(int)

    for label, _ in balanced_data_list:
        class_counts[label] += 1

    # print(class_counts)

    return balanced_data_list

import random
from collections import defaultdict


def balancing_match(data_list):
    class_counts = defaultdict(int)

    # Count the occurrences of each class
    for label, _ in data_list:
        class_counts[label] += 1

    max_samples = max(class_counts.values())

    balanced_data_list = []

    for label in class_counts:
        # Extract all samples of a class
        class_samples = [item for item in data_list if item[0] == label]
        # Calculate the number of duplicates needed
        num_needed = max_samples - class_counts[label]
        # Shuffle to add randomness
        random.shuffle(class_samples)
        # Append all original samples
        balanced_data_list.extend(class_samples)
        # Append duplicates of the samples randomly until reaching the required count
        if num_needed > 0:
            duplicates = random.choices(class_samples, k=num_needed)
            balanced_data_list.extend(duplicates)

    # Shuffle the final list to mix classes
    random.shuffle(balanced_data_list)

    # Optionally, verify the new class counts
    new_class_counts = defaultdict(int)
    for label, _ in balanced_data_list:
        new_class_counts[label] += 1

    print(new_class_counts)

    return balanced_data_list

        
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
    if __type == "train_list":
        return balancing_match(images_list)
    else:
        return images_list
    # return balancing(images_list)

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


def create_dataloader(opt, __type):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    # -------------------------------------------------------------------------------------------
    dataset = get_dataset(opt) if not opt.metadata else get_dataset_from_txt(opt, __type)
    # -------------------------------------------------------------------------------------------
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    print(sampler)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
