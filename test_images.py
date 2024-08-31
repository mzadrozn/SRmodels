import os

import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset

from scripts.efficientTransformerSR import EfficientTransformerSR
from scripts.esrtGanSR import EsrtGanSR
from scripts.srcnnSR import SrcnnSR
from scripts.srganSR import SrganSR
from scripts.appSR import AppSR
from scripts.dataloaders.dataloader import Set5Dataset, Set14Dataset, Bsd100Dataset, Urban100Dataset
from util.ycbcr_converter import convert_ycbcr_to_rgb, convert_rgb_to_ycbcr

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB, channel_axis=2)
    return s


def forward_chop(model, x, shave=15, min_size=20000):
    scale = 2
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    if h_size % 2 != 0:
        h_size += 1
    if w_size % 2 != 0:
        w_size += 1

    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def apply_shave(dim_max, shave, x_start, x_end):
    if x_start - shave < 0:
        x_end += shave
    else:
        x_start -= shave
    if x_end + shave > dim_max:
        x_start -= shave
    else:
        x_end += shave
    return x_start, x_end


def forward_image(model, x, device, piece_size=48, scale_factor=2, shave=24):
    b, c, h, w = x.size()
    tiles = []

    for i in range(0, h, piece_size):
        for j in range(0, w, piece_size):
            if j + piece_size < w:
                w_start = j
                w_end = j + piece_size
            else:
                w_start = w - piece_size
                w_end = w
            if i + piece_size < h:
                h_start = i
                h_end = i + piece_size
            else:
                h_start = h - piece_size
                h_end = h

            w_start -= shave
            w_start = max(0, w_start)
            w_end += shave
            w_end = min(w, w_end)

            h_start -= shave
            h_start = max(0, h_start)
            h_end += shave
            h_end = min(h, h_end)

            tile = x[:, :, h_start:h_end, w_start:w_end]
            tiles.append(tile)

    sr_tiles = []

    for tile in tiles:
        tile = tile.to(device)
        sr_tile_tensor = model(tile).squeeze(0)
        sr_tile = sr_tile_tensor.cpu().detach().numpy()
        sr_tiles.append(sr_tile)

    sr_image = np.empty(shape=[c, h * scale_factor, w * scale_factor])

    idx = 0
    scaled_shave = shave * scale_factor
    scaled_piece_size = piece_size * scale_factor

    for i in range(0, h, piece_size):
        for j in range(0, w, piece_size):
            if j + piece_size < w:
                w_start = j
                w_end = j + piece_size
            else:
                w_start = w - piece_size
                w_end = w
            if i + piece_size < h:
                h_start = i
                h_end = i + piece_size
            else:
                h_start = h - piece_size
                h_end = h

            sr_tile = sr_tiles[idx]

            w_start = w_start * scale_factor
            w_end = w_end * scale_factor
            h_start = h_start * scale_factor
            h_end = h_end * scale_factor

            if h_start > 0:
                if h_start > scaled_shave:
                    sr_tile = sr_tile[:, scaled_shave:, :]
                else:
                    sr_tile = sr_tile[:, h_start:, :]

            if w_start > 0:
                if w_start > scaled_shave:
                    sr_tile = sr_tile[:, :, scaled_shave:]
                else:
                    sr_tile = sr_tile[:, :, w_start:]
            h_start = max(h_start, 0)
            w_start = max(w_start, 0)

            sr_image[:, h_start:h_end, w_start:w_end] = sr_tile[:, :scaled_piece_size, :scaled_piece_size]
            idx = idx + 1

    return sr_image


class Tester():
    def __init__(self, app: AppSR, app_name, split_image=False, split_size=128, interpolate=False, scale_factor=4,
                 save_image=False, dataset_name="Set5", y_channel_model=False, shave=6):
        self.app = app
        self.split_image = split_image
        self.patch_size = split_size
        self.interpolate = interpolate
        self.scale_factor = scale_factor
        self.y_channel_model=y_channel_model
        self.shave = shave
        #self.app.loadBest()
        self.app.load()
        if app.model is None:
            self.model = app.modelG
        else:
            self.model = app.model
        self.model.eval()
        self.device = app.device
        self.test_dir = "tests" + f'\\X{scale_factor}' + "\\" + app_name + \
                        f'\\{dataset_name}' + '\\' + f'piece-size-{split_size}'
        self.save_image = save_image

    def test(self, dataset: Dataset, img_name='img'):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        idx = 1
        for img, label in dataset:
            width = img.shape[1]
            height = img.shape[2]
            file = f"{img_name}{idx}" + '.png'
            label = (np.clip(label.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
            x = img
            input = x.unsqueeze(0)
            input = input.to(self.device)

            if self.y_channel_model:
                input = convert_rgb_to_ycbcr(input*255)
                input_y = input[0, :, :].unsqueeze(0)
                input_cb = input[1, :, :].unsqueeze(0).detach().cpu().numpy()
                input_cr = input[2, :, :].unsqueeze(0).detach().cpu().numpy()
                input = input_y/255.

            if self.split_image:
                if width < self.patch_size and height < self.patch_size:
                    output = self.model(input).squeeze(0).detach().cpu().numpy()
                else:
                    output = forward_image(self.model, input, self.device, piece_size=self.patch_size,
                                           scale_factor=self.scale_factor, shave=self.shave)
            else:
                if self.y_channel_model:
                    output = self.model(input).detach().cpu().numpy()
                else:
                    output = self.model(input).squeeze(0).detach().cpu().numpy()

            output = np.clip(output, 0, 1)
            output = output * 255

            if self.y_channel_model:
                output = np.concatenate((output, input_cb, input_cr), axis=0)
                output = np.clip(convert_ycbcr_to_rgb(torch.Tensor(output)).numpy(), 0, 255).astype(np.uint8)
            else:
                output = (np.moveaxis(output, 0, -1)).astype(np.uint8)
            if self.save_image:
                img = Image.fromarray(output, 'RGB')
                if os.path.exists(self.test_dir + '\\' + file):
                    os.remove(self.test_dir + '\\' + file)
                img.save(self.test_dir + '\\' + file)
            idx = idx + 1
            psnr_calculated = compute_psnr(label, output)
            ssim_calculated = compute_ssim(label, output)
            print(f"Wartosc PSNR dla obrazu {file}, wynosi: {psnr_calculated}, ssim wynosi: {ssim_calculated}")


def test_PSNR_SSIM(app, dataset_name, dataset: Dataset, img_name='img', scale_factor=4, split_size=96):
    test_dir = "tests" + f'\\X{scale_factor}' + "\\" + app.getModelName() + \
               f'\\{dataset_name}' + '\\' + f'piece-size-{split_size}'
    idx = 1
    psnrs = []
    ssims = []
    for img, label in dataset:
        file = f"{img_name}{idx}" + '.png'
        label = (np.clip(label.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
        output = convert_rgb_to_ycbcr(np.array(Image.open(test_dir + f"\\{file}").convert("RGB"))).astype(np.uint8)
        idx = idx + 1
        output = np.expand_dims(output[4:-4, 4:-4, 0], axis=2)
        label = np.expand_dims(label[4:-4, 4:-4, 0], axis=2)
        psnrs.append(compute_psnr(label, output))
        ssims.append(compute_ssim(label, output))
    return np.mean(psnrs), np.mean(ssims)


def calculate_PSNR(scale_factor=4, split_size=128):
    config = "test"

    #[dataset, dataset_name]
    datasets = [[Set5Dataset, "Set5"], [Set14Dataset, "Set14"], [Bsd100Dataset, "BSD100"], [Urban100Dataset, "Urban100"]]

    for dataset_class, dataset_name in datasets:
        print(f"\nDATASET NAME: {dataset_name}")
        app = EfficientTransformerSR(config)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=scale_factor
        )
        psnr, ssim = test_PSNR_SSIM(app, dataset_name, dataset)
        print(f"PSNR: {psnr}, SSIM: {ssim}")

        app = SrcnnSR(config)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=scale_factor,
            interpolate=True
        )
        psnr, ssim = test_PSNR_SSIM(app, dataset_name, dataset)
        print(f"PSNR: {psnr}, SSIM: {ssim}")

        #Only generator trained
        app = SrganSR(config)
        app.pretrain = True
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=scale_factor
        )
        psnr, ssim = test_PSNR_SSIM(app, dataset_name, dataset)
        print(f"PSNR: {psnr}, SSIM: {ssim}")

        #GAN trained
        app = SrganSR(config)
        app.pretrain = False
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=scale_factor
        )
        psnr, ssim = test_PSNR_SSIM(app, dataset_name, dataset)
        print(f"PSNR: {psnr}, SSIM: {ssim}")


def create_test_images():
    config = "test"

    #[dataset, dataset_name]
    datasets = [[Set5Dataset, "Set5"], [Set14Dataset, "Set14"], [Bsd100Dataset, "BSD100"], [Urban100Dataset, "Urban100"]]

    for dataset_class, dataset_name in datasets:
        app = EfficientTransformerSR(config)
        tester = Tester(app, app_name=app.getModelName(), split_image=True, save_image=True, dataset_name=dataset_name)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=tester.scale_factor
        )
        tester.test(dataset)

        app = SrcnnSR(config)
        tester = Tester(app, app_name=app.getModelName(), interpolate=True, save_image=True, dataset_name=dataset_name,
                        y_channel_model=True)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=tester.scale_factor,
            interpolate=tester.interpolate
        )
        tester.test(dataset)

        #Only generator trained
        app = SrganSR(config)
        app.pretrain = True
        tester = Tester(app, app_name=app.getGeneratorModelName(), save_image=True, dataset_name=dataset_name)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=tester.scale_factor
        )
        tester.test(dataset)

        #GAN trained
        app = SrganSR(config)
        app.pretrain = False
        tester = Tester(app, app_name=app.getModelName(), save_image=True, dataset_name=dataset_name)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=tester.scale_factor
        )
        tester.test(dataset)

        #ESRTGAN
        app = EsrtGanSR(config)
        app.pretrain = False
        tester = Tester(app, app_name=app.getModelName(), split_image=True, save_image=True, dataset_name=dataset_name)
        dataset = dataset_class(
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            lr_scale=tester.scale_factor
        )
        tester.test(dataset)


if __name__ == '__main__':
    create_test_images()
    calculate_PSNR()

