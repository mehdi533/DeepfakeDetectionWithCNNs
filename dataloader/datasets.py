import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
from random import random

import torch
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetFromMetadata(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None, cropping=False):
        self.data_list = data_list  # List of (label, filename) tuples
        self.transform = transform
        self.cropping = cropping
        self.expanded_data_list = self._expand_data_list(data_list)

    def _expand_data_list(self, data_list):
        expanded_list = []
        nb_crops = 20 # Number of crops per image
        for label, image_path in data_list:
            for i in range(nb_crops):  
                expanded_list.append((label, image_path, i))
        return expanded_list

    def __len__(self):
        if self.cropping:
            return len(self.expanded_data_list)
        else:
            return len(self.data_list)

    def __getitem__(self, index):
        
        if self.cropping: # Optional cropping of the images
            label, image_path, crop_idx = self.expanded_data_list[index]
            image = Image.open(image_path).convert('RGB')
            # Determine the size of the crops
            width, height = image.size
            crop_width, crop_height = width // 5, height // 4

            # Calculate the coordinates for cropping
            x = (crop_idx % 5) * crop_width
            y = (crop_idx // 5) * crop_height

            image = image.crop((x, y, x + crop_width, y + crop_height))
        else:
            label, image_path = self.data_list[index]
            image = Image.open(image_path).convert('RGB') 

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(label))


def get_dataset_metadata(opt, image_list):

    # Resizing, pre-processing, and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        RandomJPEGCompression(prob=opt.compr_prob, quality=75),
        RandomBlurring(prob=opt.blur_prob, ksize=(5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DatasetFromMetadata(image_list, transform, opt.cropping)

    return dataset


# Functions for pre-processing (compression and blurring)

class RandomJPEGCompression:
    def __init__(self, prob=0, quality=75):
        self.prob = prob
        self.quality = quality

    def compress_image(self, image):
        image = np.array(image)

        # Apply JPEG compression with cv2 library
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded_image = cv2.imencode('.jpg', image, encode_param)
        compressed_image = cv2.imdecode(encoded_image, 1)

        compressed_image = Image.fromarray(compressed_image)

        return compressed_image

    def __call__(self, img):
        if random() < self.prob:
            img = self.compress_image(img)
        return img


class RandomBlurring:
    def __init__(self, prob=0.5, ksize=(5, 5)):
        self.prob = prob
        self.ksize = ksize

    def blur_image(self, image):

        # Apply blurring with cv2 library
        image = np.array(image)
        blurred_image = cv2.GaussianBlur(image, self.ksize, 0)
        
        blurred_image = Image.fromarray(blurred_image)
        
        return blurred_image

    def __call__(self, img):
        if random() < self.prob:
            img = self.blur_image(img)
        return img
