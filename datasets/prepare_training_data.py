import numpy as np
import torch
import cv2
import os
import random
import multiprocessing
import math
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
from tqdm import tqdm
from typing import Any


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


def _cubic(x: Any) -> Any:
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
        ((absx > 1) * (absx <= 2)).type_as(absx))


def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    if (scale < 1) and antialiasing:
        kernel_width = kernel_width / scale

    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)

    p = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def main():
    args = {
        "inputs_dir": "DIV2K\\DIV2K_train_HR",  # Path to input image directory.
        "output_dir": "DIV2K\\DFO2K_train_GT",  # Path to generator image directory.
        "crop_size": 400,  # Crop image size from raw image.
        "step": 200,  # Step size of sliding window.
        "thresh_size": 0,  # Threshold size. If the remaining image is less than the threshold, it will not be cropped.
        "num_workers": 6  # How many threads to open at the same time.
    }
    split_images_hr(args)

    args = {
        "inputs_dir": "DIV2K\\DFO2K_train_GT",  # Path to input image directory.
        "output_dir": "DIV2K\\DFO2K_train_LR_bicubic\\X4",  # Path to generator image directory.
        "scale": 4,  # Scale factor.
        "num_workers": 6  # How many threads to open at the same time.
    }
    split_images(args)


def split_images(args: dict):
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    num_workers = args["num_workers"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Create {output_dir} successful.")

    image_file_paths = os.listdir(inputs_dir)

    progress_bar = tqdm(total=len(image_file_paths), unit="image", desc="Split image")
    workers_pool = multiprocessing.Pool(num_workers)
    for image_file_path in image_file_paths:
        workers_pool.apply_async(worker, args=(image_file_path, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()
    print("Split image successful.")


def worker(image_file_path: str, args: dict):
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    scale = args["scale"]

    image_name, extension = os.path.splitext(os.path.basename(image_file_path))
    image = cv2.imread(os.path.join(inputs_dir, image_file_path), cv2.IMREAD_UNCHANGED)

    resize_image = image_resize(image, 1 / scale, antialiasing=True)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}{extension}"), resize_image)


def split_images_hr(args: dict):
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    num_workers = args["num_workers"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Create {output_dir} successful.")

    # Get all image paths
    image_file_paths = os.listdir(inputs_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_paths), unit="image", desc="Split image")
    workers_pool = multiprocessing.Pool(num_workers)
    for image_file_path in image_file_paths:
        workers_pool.apply_async(worker_split, args=(image_file_path, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()
    print("Split image successful.")


def worker_split(image_file_path: str, args: dict):
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    crop_size = args["crop_size"]
    step = args["step"]
    thresh_size = args["thresh_size"]

    image_name, extension = os.path.splitext(os.path.basename(image_file_path))
    image = cv2.imread(os.path.join(inputs_dir, image_file_path), cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[0:2]
    image_height_space = np.arange(0, image_height - crop_size + 1, step)
    if image_height - (image_height_space[-1] + crop_size) > thresh_size:
        image_height_space = np.append(image_height_space, image_height - crop_size)
    image_width_space = np.arange(0, image_width - crop_size + 1, step)
    if image_width - (image_width_space[-1] + crop_size) > thresh_size:
        image_width_space = np.append(image_width_space, image_width - crop_size)

    index = 0
    for h in image_height_space:
        for w in image_width_space:
            index += 1
            # Crop
            crop_image = image[h: h + crop_size, w:w + crop_size, ...]
            crop_image = np.ascontiguousarray(crop_image)
            # Save image
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_{index:04d}{extension}"), crop_image)


if __name__ == "__main__":
    main()