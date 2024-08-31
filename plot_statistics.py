import matplotlib.pyplot as plt
import numpy as np
import os
from fire import Fire
from scripts.esrtGanSR import EsrtGanSR
from scripts.srcnnSR import SrcnnSR
from scripts.srganSR import SrganSR
from scripts.efficientTransformerSR import EfficientTransformerSR
from scripts.esrtGanSR import EsrtGanSR
from collections.abc import Iterable
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from PIL import Image
import cv2

DATASET_PATH_HR = Path("datasets/Set14/Set14_HR")
DATASET_PATH_LR = Path("datasets/Set14/Set14_LR_x4")


def main(config="test"):
    plot_statistics()
    plot_psnr_ssim_comparison()
    plot_zoomed_images()


def plot_psnr(name, metric, show=False, save=False, save_filename=None, save_dir="tests\\X4\\plots\\PSNR",
              metric_name="PSNR"):
    if save_filename is None and save is True:
        save_filename = name
    if isinstance(name, list):
        for single_name, single_metric in zip(name, metric):
            plt.plot(single_metric, label=single_name)
    else:
        plt.plot(metric, label=name)
    plt.xlabel("Epoka")
    plt.ylabel(f"{metric_name}")
    plt.legend()
    if isinstance(name, list):
        plt.title(f"{metric_name} w kolejnych epokach modeli {save_filename}")
    else:
        plt.title(f"{metric_name} w kolejnych epokach modelu {name}")
    if show:
        plt.show()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}\\{save_filename}.png")
    plt.close()
    return


def plot_ssim(name, metric, show=False, save=False, save_filename=None, save_dir="tests\\X4\\plots\\SSIM",
              metric_name="SSIM"):
    if save_filename is None and save is True:
        save_filename = name
    if isinstance(name, list):
        for single_name, single_metric in zip(name, metric):
            plt.plot(single_metric, label=single_name)
    else:
        plt.plot(metric, label=name)
    plt.xlabel("Epoka")
    plt.ylabel(f"{metric_name}")
    plt.legend()
    if isinstance(name, list):
        plt.title(f"{metric_name} w kolejnych epokach modeli {save_filename}")
    else:
        plt.title(f"{metric_name} w kolejnych epokach modelu {name}")
    if show:
        plt.show()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}\\{save_filename}.png")
    plt.close()
    return


def plot_loss(name, metric, show=False, save=False, save_filename=None, save_dir="tests\\X4\\plots\\loss",
              metric_name="Wartość funkcji straty"):
    if save_filename is None and save is True:
        save_filename = name
    if isinstance(name, list):
        for single_name, single_metric in zip(name, metric):
            plt.plot(single_metric, label=single_name)
    else:
        plt.plot(metric, label=name)
    plt.xlabel("Epoka")
    plt.ylabel(f"{metric_name}")
    plt.legend()
    if isinstance(name, list):
        plt.title(f"{metric_name} w kolejnych epokach modeli {save_filename}")
    else:
        plt.title(f"{metric_name} w kolejnych epokach modelu {name}")
    if show:
        plt.show()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}\\{save_filename}.png")
    plt.close()
    return


def convertTestImage(data, save_name, x1, x2, y1, y2, show=False, zoom_loc=1):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)

    axins = zoomed_inset_axes(ax, 3, loc=zoom_loc)
    axins.imshow(data)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    if show:
        plt.show()
        return
    save_dir = "tests\\x4\\plots\\zoomed_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}\\{save_name}' + '.png')
    plt.close()


def plot_image(app, model_name, image_name, x1, x2, y1, y2, zoom_loc=1,dataset_name="Set14"):
    if model_name == "gt":
        test_dir = str(DATASET_PATH_HR)
        image = Image.open(test_dir + f'\\{image_name}.png')
    elif model_name == "lr":
        test_dir = str(DATASET_PATH_LR)
        image = Image.open(test_dir + f'\\{image_name}.png')
        width = image.size[0] * 4
        height = image.size[1] * 4
        dim = (width, height)
        image = np.array(image)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    else:
        test_dir = "tests" + '\\X4' + "\\" + app.getModelName() + \
                   f'\\{dataset_name}' + '\\' + f'piece-size-128'
        image = Image.open(test_dir + f'\\{image_name}.png')
    convertTestImage(image, f"{model_name}_{image_name}", x1, x2, y1, y2, zoom_loc=zoom_loc)


def plot_zoomed_images(config="test"):
    # Baboon
    x1, x2, y1, y2 = 140, 220, 10, 60
    # GT
    plot_image(None, "gt", "img_001_SRF_4_HR", x1, x2, y1, y2)

    # LR
    plot_image(None, "lr", "img_001_SRF_4_LR", x1, x2, y1, y2)

    app = EsrtGanSR(config)
    app.load()
    plot_image(app, "ESRTGAN", "img1", x1, x2, y1, y2)

    app = SrcnnSR(config)
    app.load()
    plot_image(app, "SRCNN", "img1", x1, x2, y1, y2)

    app = SrganSR(config)
    app.pretrain = True  # test only generator non GAN trained
    app.load()
    plot_image(app, "ResNet", "img1", x1, x2, y1, y2)

    app = SrganSR(config)
    app.pixel_weight = 0.5
    app.load()
    plot_image(app, "SRGAN", "img1", x1, x2, y1, y2)

    app = EfficientTransformerSR(config)
    app.load()
    plot_image(app, "ESRT", "img1", x1, x2, y1, y2)

    #Butterfly
    x1, x2, y1, y2 = 240, 320, 120, 220
    #GT
    plot_image(None, "gt", "img_011_SRF_4_HR", x1, x2, y1, y2)

    #LR
    plot_image(None, "lr", "img_011_SRF_4_LR", x1, x2, y1, y2)

    app = SrcnnSR(config)
    app.load()
    plot_image(app, "SRCNN", "img11", x1, x2, y1, y2)

    app = SrganSR(config)
    app.pretrain = True # test only generator non GAN trained
    app.load()
    plot_image(app, "ResNet", "img11", x1, x2, y1, y2)

    app = SrganSR(config)
    app.pixel_weight = 0.5
    app.load()
    plot_image(app, "SRGAN", "img11", x1, x2, y1, y2)

    app = EfficientTransformerSR(config)
    app.load()
    plot_image(app, "ESRT", "img11", x1, x2, y1, y2)

    #Girl
    x1, x2, y1, y2 = 140, 220, 150, 200
    #GT
    plot_image(None, "gt", "img_005_SRF_4_HR", x1, x2, y1, y2, zoom_loc=8)

    #LR
    plot_image(None, "lr", "img_005_SRF_4_LR", x1, x2, y1, y2, zoom_loc=8)

    app = SrcnnSR(config)
    app.load()
    plot_image(app, "SRCNN", "img5", x1, x2, y1, y2, zoom_loc=8)

    app = SrganSR(config)
    app.pretrain = True # test only generator non GAN trained
    app.load()
    plot_image(app, "ResNet", "img5", x1, x2, y1, y2, zoom_loc=8)

    app = SrganSR(config)
    app.pixel_weight = 0.5
    app.load()
    plot_image(app, "SRGAN", "img5", x1, x2, y1, y2, zoom_loc=8)

    app = EfficientTransformerSR(config)
    app.load()
    plot_image(app, "ESRT", "img5", x1, x2, y1, y2, zoom_loc=8)


def plot_statistics(config="test"):
    parameters_esrtgan = [(24,0.5), (24,1), (48,0.9), (48, 1)]
    for (psize, pw) in parameters_esrtgan:
        name = f"ESRTGAN-ps{psize}-pw{pw}"
        app = EsrtGanSR(config)
        app.pixel_weight = pw
        app.patchSize = psize
        app.configs['patchSize'] = psize
        app.load()
        plot_psnr(name, app.validPsnrs, show=False, save=True)
        plot_ssim(name, app.validSsims, show=False, save=True)
        plot_loss(name, app.trainLosses, show=False, save=True)

    app = SrcnnSR(config)
    app.load()
    plot_psnr("SRCNN", app.validPsnrs, show=False, save=True)
    plot_ssim("SRCNN", app.validSsims, show=False, save=True)
    plot_loss("SRCNN", app.trainLosses, show=False, save=True)

    app = SrganSR(config)
    app.pretrain = True  # test only generator non GAN trained
    app.load()
    plot_psnr("ResNet", app.validPsnrs, show=False, save=True)
    plot_ssim("ResNet", app.validSsims, show=False, save=True)
    plot_loss("ResNet", app.trainLosses, show=False, save=True)

    SRGAN_PIXELWEIGHTS = [0.1, 0.5, 0.9]
    names = []
    psnrs = []
    ssims = []
    losses = []
    for weight in SRGAN_PIXELWEIGHTS:
        app = SrganSR(config)
        app.pixel_weight = weight
        app.load()
        names.append(f"SRGAN-pw-{weight}")
        psnrs.append(app.validPsnrs)
        ssims.append(app.validSsims)
        losses.append(app.trainLosses)
    plot_psnr(names, psnrs, show=False, save=True, save_filename=f"SRGAN")
    plot_ssim(names, ssims, show=False, save=True, save_filename=f"SRGAN")
    plot_loss(names, losses, show=False, save=True, save_filename=f"SRGAN")

    app = EfficientTransformerSR(config)
    app.load()
    plot_psnr("ESRT", app.validPsnrs, show=False, save=True)
    plot_ssim("ESRT", app.validSsims, show=False, save=True)
    plot_loss("ESRT", app.trainLosses, show=False, save=True)


def plot_psnr_ssim_comparison(config='test'):
    ssims = []
    psnrs = []
    model_names = ["SRCNN", "ResNet", "SRGAN", "ESRT"]

    app = SrcnnSR(config)
    app.load()
    ssims.append(app.validSsims)
    psnrs.append(app.validPsnrs)

    app = SrganSR(config)
    app.pretrain = True  # test only generator non GAN trained
    app.load()
    ssims.append(app.validSsims)
    psnrs.append(app.validPsnrs)

    app = SrganSR(config)
    app.pixel_weight = 0.5
    app.load()
    ssims.append(app.validSsims)
    psnrs.append(app.validPsnrs)

    app = EfficientTransformerSR(config)
    app.load()
    ssims.append(app.validSsims)
    psnrs.append(app.validPsnrs)

    # print maxes
    for name, ssim, psnr in zip(model_names, ssims, psnrs):
        print(f"Maksymalne wartości PSNR {max(psnr)} oraz SSIM {max(ssim)} modelu {name}")

    # plot PSNR
    for name, ssim, psnr in zip(model_names, ssims, psnrs):
        plt.plot(psnr, label=name)
    plt.legend()
    plt.xlabel("Epoka")
    plt.ylabel("PSNR")
    plt.legend()
    save_dir = "tests\\x4\\plots\\comparison"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}\\PSNR' + '.png')
    plt.close()

    # plot SSIM
    for name, ssim, psnr in zip(model_names, ssims, psnrs):
        plt.plot(ssim, label=name)
    plt.legend()
    plt.xlabel("Epoka")
    plt.ylabel("SSIM")
    plt.legend()
    save_dir = "tests\\x4\\plots\\comparison"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}\\SSIM' + '.png')
    plt.close()


if __name__ == '__main__':
    Fire(main)
