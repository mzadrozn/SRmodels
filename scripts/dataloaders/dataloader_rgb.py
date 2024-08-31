from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.v2 import functional as F
import random
from torch.nn import functional as Func
import torch


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

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, crop_size=48, rotation=False, transform=None, is_training=True, lr_scale=2,
                 interpolate=False):
        self.root_dir = root_dir
        self.interpolate = interpolate
        self.is_training = is_training
        self.crop_size = crop_size
        self.rotation = rotation
        self.transform = transform
        self.lr_scale = lr_scale
        root_dir = str(root_dir.resolve().absolute())
        self.train_hr_image_path = root_dir + '\\DIV2K_train_HR'
        self.train_lr_image_path = root_dir + '\\DIV2K_train_LR_bicubic\\X' + str(lr_scale)
        self.valid_hr_image_path = root_dir + '\\DIV2K_valid_HR'
        self.valid_lr_image_path = root_dir + '\\DIV2K_valid_LR_bicubic\\X' + str(lr_scale)

    def __len__(self):
        return 800 if self.is_training else 100

    def __getitem__(self, idx):
        if self.is_training:
            img_path = self.train_lr_image_path + '\\' + \
                       str(idx + 1).zfill(4) + 'x' + str(self.lr_scale) + '.png'
            label_path = self.train_hr_image_path + \
                         '\\' + str(idx + 1).zfill(4) + '.png'
        else:
            img_path = self.valid_lr_image_path + '\\' + \
                       str(idx + 801).zfill(4) + 'x' + str(self.lr_scale) + '.png'
            label_path = self.valid_hr_image_path + \
                         '\\' + str(idx + 801).zfill(4) + '.png'

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        W = img.size()[1]
        H = img.size()[2]
        if self.is_training:
            Ws = np.random.randint(0, W - self.crop_size + 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size + 1, 1)[0]
        else:
            Ws = int(W / 2) - int(self.crop_size / 2)
            Hs = int(H / 2) - int(self.crop_size / 2)

        img_crop = img[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
        label_crop = label[:, Ws * self.lr_scale:(Ws + self.crop_size) *
                                                 self.lr_scale,
                     Hs * self.lr_scale: (Hs + self.crop_size) * self.lr_scale]
        if self.is_training:
            if random.randint(0, 1) == 1:
                img_crop = F.hflip(img_crop)
                label_crop = F.hflip(label_crop)

            if random.randint(0, 1) == 1:
                img_crop = F.rotate(img_crop, 90)
                label_crop = F.rotate(label_crop, 90)

        if self.interpolate:
            img_crop = Func.interpolate(img_crop.unsqueeze(0), scale_factor=self.lr_scale, mode='bicubic').squeeze(0)

        return img_crop, label_crop


class Set5Dataset(Dataset):
    def __init__(self, root_dir='datasets\\Set5', transform=None, lr_scale=4, interpolate=False):
        self.root_dir = root_dir
        self.transform = transform
        self.lr_scale = lr_scale
        self.img_names = ['baby', 'bird', 'butterfly', 'head', 'woman']
        self.hr_image_path = root_dir + '\\Set5_HR'
        self.lr_image_path = root_dir + f'\\Set5_LR_x{lr_scale}'
        self.interpolate = interpolate

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.lr_image_path + '\\' + img_name + '.png'
        label_path = self.hr_image_path + '\\' + img_name + '.png'
        img = Image.open(img_path)
        img = np.array(img)
        label = np.array(Image.open(label_path))

        if self.interpolate:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = img.permute(2,0,1).unsqueeze(0)
            img = Func.interpolate(img, scale_factor=self.lr_scale, mode='bicubic')\
                .squeeze(0).permute(1,2,0).numpy()

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label


class Set14Dataset(Dataset):
    def __init__(self, root_dir='datasets\\Set14', transform=None, lr_scale=4, interpolate=False):
        self.root_dir = root_dir
        self.interpolate = interpolate
        self.transform = transform
        self.lr_scale = lr_scale
        self.hr_image_path = root_dir + '\\Set14_HR'
        self.lr_image_path = root_dir + f'\\Set14_LR_x{lr_scale}'

    def __len__(self):
        return 14

    def __getitem__(self, idx):
        img_name = f"img_{idx+1:03d}_SRF_{self.lr_scale}_LR"
        img_path = self.lr_image_path + '\\' + img_name + '.png'
        label_name = f"img_{idx+1:03d}_SRF_{self.lr_scale}_HR"
        label_path = self.hr_image_path + '\\' + label_name + '.png'
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        label = np.array(Image.open(label_path).convert("RGB"))

        if self.interpolate:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = img.permute(2,0,1).unsqueeze(0)
            img = Func.interpolate(img, scale_factor=self.lr_scale, mode='bicubic')\
                .squeeze(0).permute(1,2,0).numpy()

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label


class Bsd100Dataset(Dataset):
    def __init__(self, root_dir='datasets\\BSD100', transform=None, lr_scale=4, interpolate=False):
        self.root_dir = root_dir
        self.interpolate = interpolate
        self.transform = transform
        self.lr_scale = lr_scale
        self.hr_image_path = root_dir + f'\\image_SRF_{lr_scale}'
        self.lr_image_path = root_dir + f'\\image_SRF_{lr_scale}'

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        img_name = f"img_{idx + 1:03d}_SRF_{self.lr_scale}_LR"
        img_path = self.lr_image_path + '\\' + img_name + '.png'
        label_name = f"img_{idx + 1:03d}_SRF_{self.lr_scale}_HR"
        label_path = self.hr_image_path + '\\' + label_name + '.png'
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        label = np.array(Image.open(label_path).convert("RGB"))

        if self.interpolate:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = Func.interpolate(img, scale_factor=self.lr_scale, mode='bicubic') \
                .squeeze(0).permute(1, 2, 0).numpy()

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label


class Urban100Dataset(Dataset):
    def __init__(self, root_dir='datasets\\Urban100', transform=None, lr_scale=4, interpolate=False):
        self.root_dir = root_dir
        self.interpolate = interpolate
        self.transform = transform
        self.lr_scale = lr_scale
        self.hr_image_path = root_dir + f'\\image_SRF_{lr_scale}'
        self.lr_image_path = root_dir + f'\\image_SRF_{lr_scale}'

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        img_name = f"img_{idx + 1:03d}_SRF_{self.lr_scale}_LR"
        img_path = self.lr_image_path + '\\' + img_name + '.png'
        label_name = f"img_{idx + 1:03d}_SRF_{self.lr_scale}_HR"
        label_path = self.hr_image_path + '\\' + label_name + '.png'
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        label = np.array(Image.open(label_path).convert("RGB"))

        if self.interpolate:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = Func.interpolate(img, scale_factor=self.lr_scale, mode='bicubic') \
                .squeeze(0).permute(1, 2, 0).numpy()

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label