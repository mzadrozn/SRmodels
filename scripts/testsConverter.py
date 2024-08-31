import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
from pathlib import Path
from PIL import Image
from numpy import asarray
from pathlib import Path
from skimage.metrics import structural_similarity
import cv2


def convertTestImage(data, save_name, x1, x2, y1, y2, show=False):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)

    axins = zoomed_inset_axes(ax, 3, loc=1)
    axins.imshow(data)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    patch, pp1,pp2 = mark_inset(ax, axins, loc1=2, loc2=3)
    pp1.loc1 = 2
    pp1.loc2 = 3

    pp2.loc1 = 3
    pp2.loc2 = 2

    if show:
        plt.show()
        return
    #plt.show()
    plt.savefig(f'converted_images\\{save_name}' + '.png')
    plt.close()


def convertTestImages(paths, models, name, x1, x2, y1, y2):
    gt_image_path = ""
    for img, model in zip(paths, models):
        if model == 'gt':
            gt_image_path = img

    gt_image = Image.open(gt_image_path)
    gt_image = asarray(gt_image)

    for img, model in zip(paths, models):
        image = Image.open(img)
        if model == 'lr':
            width = image.size[0] * 4
            height = image.size[1] * 4
            dim = (width, height)
            image = np.array(image)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
        else:
            image = asarray(image)
        psnr = cv2.PSNR(image, gt_image)
        print(f"Wartosc PSNR dla obrazu {name} dla modelu {model} wynosi: {psnr}")
        convertTestImage(image, f'{name}_' + model, x1, x2, y1, y2)


if __name__ == "__main__":
    dataset_path = Path("C:/Users/zadro/OneDrive/Desktop/SRmodels/datasets/Set14")
    tests_path = Path("C:/Users/zadro/OneDrive/Desktop/SRmodels/tests/X4")
    model_names = ['ESRT', 'ResNet', 'SRCNN', 'SRGAN', 'gt', 'lr']
    baboon_img_name = "img_001.png"
    baboon_paths = []
    for model in model_names:
        if model == 'gt':
            baboon_paths.append(str(dataset_path / "Set14_HR" / baboon_img_name))
        elif model == 'lr':
            baboon_paths.append(str(dataset_path / "Set14_LR_x4" / baboon_img_name))
        else:
            baboon_paths.append(str(tests_path))
    image_path_srgan = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRmodels\\tests\\X4\\SRGAN-lr0.0002-flipTrue-psize96\\Set14-post-gan\\img_001.png"
    image_path_esrt = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRmodels\\tests\\X4\\ETSR-lr0.0002-flipTrue-psize80\\Set14\\" \
                      "piece-size-96\\img_001.png"
    image_path_srcnn = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRmodels\\tests\\X4\\SRCNN-lr0.0002-flipFalse\\Set14\\" \
                       "img_001.png"
    image_path_resnet = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\tests\\X4\\SRGAN-lr0.0002-flipTrue-psize96" \
                        "\\Set14-pre-gan\\img_001.png"
    image_path_gt = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRmodels\\datasets\\Set14\\Set14_HR\\img_001.png"
    image_path_lr = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRmodels\\datasets\\Set14\\Set14_LR_x4\\img_001.png"
    #baboon zoom coords
    x1, x2, y1, y2 = 140, 220, 10, 60
    model_names = ['ESRT', 'ResNet', 'SRCNN', 'SRGAN', 'gt', 'lr']
    baboon_paths = list([image_path_esrt, image_path_resnet, image_path_srcnn, image_path_srgan, image_path_gt,
                         image_path_lr])
    convertTestImages(baboon_paths, model_names, 'baboon', x1, x2, y1, y2)


    image_path_srgan = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\tests\\X4\\SRGAN-lr0.0002-flipTrue-psize96\\Set14-post-gan\\img_005.png"
    image_path_esrt = "C:\\Users\\zadro\\OneDrive\\Desktop\\ESRT\\tests\\X4\\ETSR-lr0.0002-flipTrue-psize80\\Set14\\" \
                      "piece-size-96\\img_005.png"
    image_path_srcnn = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRCNN\\tests\\X4\\SRCNN-lr0.0002-flipFalse\\Set14\\" \
                       "img_005.png"
    image_path_resnet = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\tests\\X4\\SRGAN-lr0.0002-flipTrue-psize96" \
                        "\\Set14-pre-gan\\img_005.png"
    image_path_gt = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\datasets\\Set14\\Set14_HR\\img_005.png"
    image_path_lr = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\datasets\\Set14\\Set14_LR_x4\\img_005.png"

    girl_paths = list([image_path_esrt, image_path_resnet, image_path_srcnn, image_path_srgan, image_path_gt,
                         image_path_lr])
    #girl zoom coords
    x1, x2, y1, y2 = 95, 145, 200, 250
    convertTestImages(girl_paths, model_names, 'girl', x1, x2, y1, y2)


    image_path_srgan = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\tests\\X4\\SRGAN-lr0.0002-flipTrue-psize96\\Set14-post-gan\\img_011.png"
    image_path_esrt = "C:\\Users\\zadro\\OneDrive\\Desktop\\ESRT\\tests\\X4\\ETSR-lr0.0002-flipTrue-psize84\\Set14\\" \
                      "piece-size-96\\img_011.png"
    image_path_srcnn = "C:\\Users\\zadro\\OneDrive\\Desktop\\SRCNN\\tests\\X4\\SRCNN-lr0.0002-flipFalse\\Set14\\" \
                       "img_011.png"
    image_path_resnet = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\tests\\X4\\SRGAN-lr0.0002-flipTrue-psize96" \
                        "\\Set14-pre-gan\\img_011.png"
    image_path_gt = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\datasets\\Set14\\Set14_HR\\img_011.png"
    image_path_lr = "C:\\Users\\zadro\\OneDrive\\Desktop\\Fast-SRGAN\\datasets\\Set14\\Set14_LR_x4\\img_011.png"
    butterfly_paths = list([image_path_esrt, image_path_resnet, image_path_srcnn, image_path_srgan, image_path_gt,
                         image_path_lr])
    #girl zoom coords
    x1, x2, y1, y2 = 460, 500, 325, 365
    convertTestImages(butterfly_paths, model_names, 'butterfly', x1, x2, y1, y2)

