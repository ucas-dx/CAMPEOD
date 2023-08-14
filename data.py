#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/28 20:33
# @Author  : Denxun
# @FileName: data.py
# @Software: PyCharm
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage.util import img_as_ubyte
from torch.optim.lr_scheduler import StepLR
from skimage import io
import torchvision.transforms as transforms
import tqdm
import random
from torchvision import models
import matplotlib.pyplot as plt

# import skimage
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# data = torch.ones(2, 1, 512, 512)
# from torchcontrib.optim import SWA
import torchvision
class CellSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'segmentations')
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = io.imread(img_name)
        mask_name = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = io.imread(mask_name)
        image = img_as_ubyte(image)
        mask = img_as_ubyte(mask)
        sample = {'image': image, 'mask': mask}
        if self.image_transform:
            sample['image'] = self.image_transform(sample['image'])
        if self.mask_transform:
            sample['mask'] = self.mask_transform(sample['mask'])
        return sample

image_transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([512, 512]),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    # transforms.Normalize(0.40081004856666735,0.0877802242064718),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])
image_transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    # transforms.Normalize(0.3579441845027396,0.10539796972687705),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])
mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    lambda x: torch.as_tensor(x, dtype=torch.float32)
])

# root_dir_train = r"PATH FOR YOUR DATA"
# root_dir_Val = r'PATH FOR YOUR DATA'
# root_dir_test = r'PATH FOR YOUR DATA'
root_dir_train = r"test_pad_1c"
root_dir_Val = r'test_pad_1c'
root_dir_test = r'test_pad_1c'


cell_dataset_train = CellSegmentationDataset(root_dir_train, image_transform=image_transform_train, mask_transform=mask_transform)
cell_dataset_test = CellSegmentationDataset(root_dir_test, image_transform=image_transform_train, mask_transform=mask_transform)
cell_dataset_val = CellSegmentationDataset(root_dir_Val, image_transform=image_transform_train, mask_transform=mask_transform)

train_loader = torch.utils.data.DataLoader(cell_dataset_train, batch_size=1, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(cell_dataset_test, batch_size=1, shuffle=False, num_workers=0)
Val_loader = torch.utils.data.DataLoader(cell_dataset_val, batch_size=1, shuffle=False, num_workers=0)