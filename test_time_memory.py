import os

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn import functional as Func
from torch.utils.data import Dataset
from typing import Tuple
from torch.profiler import profile, record_function, ProfilerActivity

from scripts.efficientTransformerSR import EfficientTransformerSR
from scripts.esrtGanSR import EsrtGanSR
from scripts.srcnnSR import SrcnnSR
from scripts.srganSR import SrganSR
from scripts.appSR import AppSR
from scripts.dataloaders.dataloader import Set5Dataset, Urban100Dataset

TEST_SIZE = 1000
SAMPLE_INDEX = 4


def time_test(dataset: Dataset, image_index: int, app: AppSR, test_size=100, image_size: Tuple[int, int] = (24, 24)):
    if app.model is not None:
        model = app.model
    else:
        model = app.modelG
    model.eval()
    device = app.device

    test_img, _ = dataset[image_index]
    test_img = test_img.unsqueeze(0)
    test_img = test_img[:, :, :image_size[0], :image_size[1]]
    test_img = test_img.to(device)
    t0 = time.time()
    for _ in range(test_size):
        model(test_img)

    t1 = time.time()
    total_n = t1 - t0
    mean_t = total_n / test_size * 1000  # konwersja na ms
    return mean_t


def mem_test(dataset: Dataset, image_index: int, app: AppSR, image_size: Tuple[int, int] = (24, 24)):
    torch.cuda.reset_peak_memory_stats()
    test_img, _ = dataset[image_index]
    test_img = test_img.unsqueeze(0)
    test_img = test_img[:, :, :image_size[0], :image_size[1]]
    app.load()
    if app.model is not None:
        model = app.model
    else:
        model = app.modelG
    model.eval()
    device = app.device
    test_img = test_img.to(device)
    model(test_img)
    return (torch.cuda.max_memory_allocated(device=device)) / 1000000000  # return in GB


def perform_time_test(config="test"):
    sr_app = SrcnnSR(config)
    sr_app.load()
    data = Set5Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        interpolate=True
    )
    print(f"Warmup SRCNN: {time_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, test_size=TEST_SIZE)}")
    print(f"SRCNN: {time_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, test_size=TEST_SIZE)}")

    sr_app = EfficientTransformerSR(config)
    sr_app.load()
    data = Set5Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    print(f"Warmup ESRT: {time_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, test_size=TEST_SIZE)}")
    print(f"ESRT: {time_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, test_size=TEST_SIZE)}")

    # ResNet
    sr_app = SrganSR(config)
    sr_app.pretrain = True
    sr_app.load()
    data = Set5Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    print(f"Warmup ResNet: {time_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, test_size=TEST_SIZE)}")
    print(f"ResNet: {time_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, test_size=TEST_SIZE)}")


def perform_mem_test(config="test", image_sizes=range(30, 160, 10), sleep_time=0.2):
    fig1, ax1 = plt.subplots()

    sr_app = SrcnnSR(config)
    data = Urban100Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        interpolate=True
    )
    results = []
    for image_size in image_sizes:
        result = mem_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, image_size=(image_size, image_size))
        results.append(result)
        torch.cuda.empty_cache()
        time.sleep(sleep_time) #Give time to free previous memory
    ax1.plot(image_sizes, results, label="SRCNN")

    sr_app = SrganSR(config)
    data = Urban100Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    results = []
    for image_size in image_sizes:
        result = mem_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, image_size=(image_size, image_size))
        results.append(result)
        torch.cuda.empty_cache()
        time.sleep(sleep_time)

    ax1.plot(image_sizes, results, label="ResNet")

    sr_app = EfficientTransformerSR(config)
    data = Urban100Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    results = []
    for image_size in image_sizes:
        result = mem_test(dataset=data, image_index=SAMPLE_INDEX, app=sr_app, image_size=(image_size, image_size))
        results.append(result)
        torch.cuda.empty_cache()
        time.sleep(sleep_time)

    ax1.plot(image_sizes, results, label="ESRT")

    ax1.set_yscale("log")
    ax1.set_yticks([0.1, 1, 10])
    ax1.legend()
    plt.xlabel("Wielkość obrazu (wysokość oraz szerokość)")
    plt.ylabel(f"GB zarezerwowanej pamięci")

    save_dir = "tests\\X4\\plots\\memory"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}\\mem_comp.png")


if __name__ == "__main__":
    perform_mem_test()
    perform_time_test()
