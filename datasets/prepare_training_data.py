import numpy as np
import torch
import cv2
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torch.nn.functional as nnf
from util.ycbcr_converter import convert_rgb_to_ycbcr
from os import listdir
from os.path import isfile, join
from torch.nn import functional as Func
from util.ycbcr_converter import convert_ycbcr_to_rgb, convert_rgb_to_ycbcr


class DFO2KDataset(Dataset):
    def __init__(self, root_dir, crop_size=24, rotation=False, transform=None, is_training=True, lr_scale=2,
                 full_scale=False, interpolate=False, y_channel_only=False):
        self.root_dir = root_dir
        self.is_training = is_training
        self.rotation = rotation
        self.transform = transform
        self.patch_size = crop_size
        self.lr_scale = lr_scale
        self.full_scale = full_scale
        self.interpolate = interpolate
        self.y_channel_only = y_channel_only
        root_dir = str(root_dir.resolve().absolute())
        self.train_hr_image_path = root_dir + '\\DFO2K_train_GT'
        self.train_lr_image_path = root_dir + '\\DFO2K_train_LR_bicubic\\X' + str(lr_scale)
        self.valid_hr_image_path = root_dir + '\\DFO2K_train_GT'
        self.valid_lr_image_path = root_dir + '\\DFO2K_train_LR_bicubic\\X' + str(lr_scale)
        self.file_names = [f for f in listdir(self.train_hr_image_path) if isfile(join(self.train_hr_image_path, f))]

    def __len__(self):
        return 40000 if self.is_training else 9080

    def __getitem__(self, idx):
        if self.is_training:
            file_name = self.file_names[idx]
            img_path = self.train_lr_image_path + '\\' + file_name
            label_path = self.train_hr_image_path + '\\' + file_name
            img = read_image(img_path)
            label = read_image(label_path)

        else:
            idx = idx + 40000
            file_name = self.file_names[idx]

            img_path = self.valid_lr_image_path + '\\' + file_name
            label_path = self.valid_hr_image_path + '\\' + file_name

            img = read_image(img_path)
            label = read_image(label_path)

        W = img.size()[1]
        H = img.size()[2]

        if self.y_channel_only:
            img = convert_rgb_to_ycbcr(img)[0, :, :].unsqueeze(0)
            label = convert_rgb_to_ycbcr(label)[0, :, :].unsqueeze(0)

        img = img/255.
        label = label/255.

        if self.is_training:
            Ws = np.random.randint(0, W - self.patch_size + 1, 1)[0]
            Hs = np.random.randint(0, H - self.patch_size + 1, 1)[0]
        else:
            Ws = int(W/2) - int(self.patch_size/2)
            Hs = int(H/2) - int(self.patch_size/2)

        img_out = img[:, Ws:Ws+(self.patch_size), Hs:Hs+(self.patch_size)]
        label_out = label[:, Ws*self.lr_scale:(Ws+self.patch_size) * self.lr_scale, Hs*self.lr_scale:(Hs+self.patch_size)*self.lr_scale]
        if self.is_training:
            if random.randint(0, 1) == 1:
                img_out = F.hflip(img_out)
                label_out = F.hflip(label_out)

            if random.randint(0, 1) == 1:
                img_out = F.rotate(img_out, 90)
                label_out = F.rotate(label_out, 90)

            if random.randint(0, 1) == 1:
                img_out = F.vflip(img_out)
                label_out = F.vflip(label_out)

        if self.interpolate:
            img_out = Func.interpolate(img_out.unsqueeze(0), scale_factor=self.lr_scale, mode='bicubic').squeeze(0)

        return img_out, label_out


def prepare(train_lr_image_path, train_hr_image_path, lr_scale, patch_size):
    it = 0
    patch_size = patch_size + 4
    out_dir_hr = Path(str(train_hr_image_path) + "_prepared")
    out_dir_lr = Path(str(train_lr_image_path) + "_prepared")
    if not os.path.exists(str(out_dir_hr)):
        os.makedirs(str(out_dir_hr))

    if not os.path.exists(str(out_dir_lr)):
        os.makedirs(str(out_dir_lr))

    if not os.path.exists(out_dir_lr / f'X{lr_scale}'):
        os.makedirs(out_dir_lr / f'X{lr_scale}')

    for i in range(800):
        img_path = train_lr_image_path / f'X{lr_scale}' / (str(i + 1).zfill(4) + 'x' + str(lr_scale) + '.png')
        label_path = train_hr_image_path / (str(i + 1).zfill(4) + '.png')
        img = read_image(str(img_path))
        label = read_image(str(label_path))
        W = img.size()[1]
        H = img.size()[2]
        for z in range(20):
            it += 1
            Ws = np.random.randint(0, W - patch_size + 1, 1)[0]
            Hs = np.random.randint(0, H - patch_size + 1, 1)[0]
            img_out = img[:, Ws:Ws + patch_size, Hs:Hs + patch_size].numpy()
            label_out = label[:, Ws * lr_scale:(Ws + patch_size) * lr_scale, Hs * lr_scale:(Hs + patch_size) * lr_scale].numpy()

            out_path_hr = out_dir_hr / (str(it).zfill(4) + '.npy')
            out_path_lr = out_dir_lr / f'X{lr_scale}' / (str(it).zfill(4) + 'x' + str(lr_scale) + '.npy')
            with open(out_path_hr, 'wb') as f:
                np.save(f, label_out / 255)

            with open(out_path_lr, 'wb') as f:
                np.save(f, img_out / 255)


if __name__ == '__main__':
    path = Path('../').resolve() / "datasets" / "DIV2K"
    prepare(path / "DIV2K_train_LR_bicubic", path / "DIV2K_train_HR", 4, 24)