import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import torch
from torch.utils.data import ConcatDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Custom dataset class to load using metadata...
class DatasetFromMetadata(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list  # List of (label, filename) tuples
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Get label and filename from the list
        label, image_path = self.data_list[index]
        # Load image
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(label))



def get_dataset_metadata(opt, image_list):
    # Define transformations if needed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DatasetFromMetadata(image_list, transform)

    if opt.blurring:
        print("Blurring Active")
        transform_blur = transforms.Compose([transforms.Lambda(lambda img: blurring(img, opt)), transform])
        dataset = ConcatDataset([dataset, DatasetFromMetadata(image_list, transform_blur)])
    
    if opt.compression:
        print("Compression Active")
        transform_comp = transforms.Compose([transforms.Lambda(lambda img: compressing(img, opt)), transform])
        dataset = ConcatDataset([dataset, DatasetFromMetadata(image_list, transform_comp)])

    return dataset


def compressing(img, opt):
    img = np.array(img)

    method = sample_discrete(opt.jpg_method)
    qual = sample_discrete(opt.jpg_qual)
    img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def blurring(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)
    
    return Image.fromarray(img)


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


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
