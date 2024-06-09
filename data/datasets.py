import cv2
import io
import numpy as np
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from random import random, choice

import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from scipy.ndimage.filters import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# class DatasetFromMetadata(torch.utils.data.Dataset):
#     def __init__(self, data_list, transform=None):
#         self.data_list = data_list  # List of (label, filename) tuples
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):

#         # Get label and filename from the list and load image
#         label, image_path = self.data_list[index]
#         image = Image.open(image_path).convert('RGB') 

#         # Apply transformations if any
#         if self.transform:
#             image = self.transform(image)

#         return image, torch.tensor(int(label))


class DatasetFromMetadata(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list  # List of (label, filename) tuples
        self.transform = transform
        self.expanded_data_list = self._expand_data_list(data_list)

    def _expand_data_list(self, data_list):
        expanded_list = []
        for label, image_path in data_list:
            for i in range(20):  # Create 20 entries per image
                expanded_list.append((label, image_path, i))
        return expanded_list

    def __len__(self):
        return len(self.expanded_data_list)

    def __getitem__(self, index):
        # Get label, filename, and crop index from the expanded list
        label, image_path, crop_idx = self.expanded_data_list[index]
        image = Image.open(image_path).convert('RGB')

        # Determine the size of the crops
        width, height = image.size
        crop_width, crop_height = width // 5, height // 4

        # Calculate the coordinates for cropping
        x = (crop_idx % 5) * crop_width
        y = (crop_idx // 5) * crop_height

        # Crop the image
        crop = image.crop((x, y, x + crop_width, y + crop_height))

        # Apply transformations if any
        if self.transform:
            crop = self.transform(crop)

        return crop, torch.tensor(int(label))


def get_dataset_metadata(opt, image_list):

    # Basic resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        RandomJPEGCompression(prob=opt.compr_prob, quality=75),
        # transforms.Lambda(lambda img: blurring(img, blur_prob=opt.blur_prob)),
        RandomBlurring(prob=opt.blur_prob, ksize=(5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DatasetFromMetadata(image_list, transform)

    # Optional additional data (blurring)
    if opt.blurring:
        print("Blurring Active")
        transform_blur = transforms.Compose([transforms.Lambda(lambda img: blurring(img)), transform])
        dataset = ConcatDataset([dataset, DatasetFromMetadata(image_list, transform_blur)])
    
    # Optional additional data (compression)
    if opt.compression:
        print("Compression Active")
        transform_comp = transforms.Compose([transforms.Lambda(lambda img: compressing(img)), transform])
        dataset = ConcatDataset([dataset, DatasetFromMetadata(image_list, transform_comp)])

    return dataset

# Functions for augmentation (compression and blurring)

def compressing(image, jpg_method='cv2', compr_prob=0, jpg_qual=[75]):
    img = np.array(image)

    if random() < compr_prob:
        method = sample_discrete(jpg_method)
        qual = sample_discrete(jpg_qual)
        img = jpeg_from_key(img, qual, method)
        return Image.fromarray(img)
    else:
        return image


def blurring(image, blur_prob=0, blur_sig=0.5):
    img = np.array(image)

    if random() < blur_prob:
        sig = sample_continuous(blur_sig)
        gaussian_blur(img, sig)
        return Image.fromarray(img)
    else:
        return image


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def compress_image(image, quality=75):
    image = np.array(image)
    # Apply JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    # Decode the image back to OpenCV format
    compressed_image = cv2.imdecode(encoded_image, 1)
    # Convert back to PIL image
    compressed_image = Image.fromarray(compressed_image)
    return compressed_image


class RandomJPEGCompression:
    def __init__(self, prob=0.5, quality=50):
        self.prob = prob
        self.quality = quality

    def __call__(self, img):
        if random() < self.prob:
            img = compress_image(img, quality=self.quality)
        return img

def blur_image(image, ksize=(5, 5)):
    """
    Apply blurring to an image.

    Args:
        image (PIL.Image): The input image.
        ksize (tuple): Kernel size for the blurring.

    Returns:
        PIL.Image: The blurred image.
    """
    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to OpenCV format (NumPy array)
    image = np.array(image)
    
    # Apply blurring
    blurred_image = cv2.GaussianBlur(image, ksize, 0)
    
    # Convert back to PIL image
    blurred_image = Image.fromarray(blurred_image)
    
    return blurred_image

class RandomBlurring:
    def __init__(self, prob=0.5, ksize=(5, 5)):
        """
        Initialize the RandomBlurring transformation.

        Args:
            prob (float): Probability of applying the blurring.
            ksize (tuple): Kernel size for the blurring.
        """
        self.prob = prob
        self.ksize = ksize

    def __call__(self, img):
        """
        Apply the transformation to an image.

        Args:
            img (PIL.Image): The input image.

        Returns:
            PIL.Image: The transformed image.
        """
        if random() < self.prob:
            img = blur_image(img, ksize=self.ksize)
        return img
