import math
import cv2
import torch
import numpy as np

from torch import nn
from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.SRGAN.resnet import SRResNet
from models.SRGAN.discriminator import Discriminator
from datasets.prepare_training_data import DFO2KDataset
from .configParser import ConfigParser
from .utils import *
from scripts.appSR import AppSR
from models.SRGAN.loss import VGGLoss
from torch.cuda import amp
from models.ESRT.esrt import ESRT

MODEL_NAME = "ESRTGAN"


class EsrtGanSR(AppSR):
    def __init__(self, configs="train"):
        super().__init__()
        title("Initialize")
        self.configs = None
        self.epoch = None
        self.initConfigs(configs)
        self.initParams()

    def initConfigs(self, configs):
        self.configs = configs or self.configs
        self.configs = ConfigParser(MODEL_NAME, self.configs).content
        if self.configs["usegpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            warn('Using CPU.')

    def trainEpochs(self, start, end):
        self.epoch = start
        self.endEpoch = end
        if self.pretrain:
            for i in range(start, end):
                self.epoch = i
                trainLoss, train_dLoss, mean_psnr, mean_ssim = self.epochAction("train", self.trainloader,
                                                                                generatorOnly=True)
                self.trainLosses.append(trainLoss)
                self.trainPsnrs.append(mean_psnr)
                self.trainSsims.append(mean_ssim)
                validLoss, valid_dLoss, mean_psnr, mean_ssim = self.epochAction("valid", self.validloader,
                                                                                generatorOnly=True)
                self.validLosses.append(validLoss)
                self.learningRates.append(self.learningRate)
                self.validSsims.append(mean_ssim)
                self.validPsnrs.append(mean_psnr)
                self.schedulerG.step()
                if (i + 1) % self.configs["saveEvery"] == 0:
                    self.save()
        else:
            for epoch in range(start, end):
                self.epoch = epoch
                trainLoss, train_dLoss, mean_psnr, mean_ssim = self.epochAction("train", self.trainloader)
                self.trainLosses.append(trainLoss)
                validLoss, train_dLoss, mean_psnr, mean_ssim = self.epochAction("valid", self.validloader)
                info(f'Valid loss: {round(validLoss, 3)}')
                self.validLosses.append(validLoss)
                self.learningRates.append(self.learningRate)
                self.validSsims.append(mean_ssim)
                self.validPsnrs.append(mean_psnr)
                self.schedulerG.step()
                self.schedulerD.step()
                if validLoss < self.bestValidLoss:
                    self.bestValidLoss = validLoss
                    [best.unlink() for best in
                     getFiles(self.getCheckpointFolder(), "best*.pth")]  # remove last best pth
                    self.save(f"bestEpoch{epoch + 1}.pth")
                    info(f"save best model, valid loss {round(validLoss, 4)}")
                if (epoch + 1) % self.configs["saveEvery"] == 0:
                    self.save()

    @property
    def learningRate(self):
        return self.optimizerG.param_groups[0]['lr']

    def modelForward(self, x, y):
        device = self.device
        x, y = map(lambda t: rearrange(t.to(device), 'b p c h w -> (b p) c h w'), (x, y))
        out = self.modelG(x)
        loss = self.l1_loss(out, y)
        return x, y, out, loss

    def epochAction(self, action, loader, generatorOnly=False, discriminatorOnly=False):
        psnr_list, ssim_list = [], []
        running_results = {'iterations': 0, 'd_loss': 0,
                           "g_loss": 0, "d_score": 0, "g_score": 0}
        isBackward = True if action == "train" else False
        GradSelection = Grad if isBackward else torch.no_grad
        totalLoss, totalCorrect, totalLen, total_dLoss = 0, 0, 0, 0
        batchLoader = tqdm(loader)
        if isBackward:
            self.modelG.train()
            self.modelD.train()
        else:
            self.modelG.eval()
            self.modelD.eval()
        with GradSelection():
            for x, y in batchLoader:
                device = self.device
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)
                b_size = x.shape[0]
                if discriminatorOnly:
                    sr_imgs = self.modelG(x)

                    discriminated = self.modelD(torch.cat((y, sr_imgs.detach()), 0))
                    proper = torch.cat((torch.ones(b_size), torch.zeros(b_size)), 0).to(device)

                    d_loss = self.adv_loss(discriminated, proper)
                    self.optimizerD.zero_grad()
                    if isBackward:
                        d_loss.backward()
                        self.optimizerD.step()

                elif not generatorOnly:
                    real_label = torch.full([b_size, 1], 1.0, dtype=torch.float, device=device)
                    fake_label = torch.full([b_size, 1], 0.0, dtype=torch.float, device=device)

                    for d_parameter in self.modelD.parameters():
                        d_parameter.requires_grad = False

                    self.modelG.zero_grad(set_to_none=True)

                    sr_imgs = self.modelG(x)
                    # Update generator GAN
                    with amp.autocast():
                        sr_discriminated = self.modelD(sr_imgs).squeeze()
                        if self.pixel_weight != 1:
                            vgg_loss = (1-self.pixel_weight) * self.vgg_loss(sr_imgs, y)
                        #pixel_loss = self.pixel_weight * self.l1_loss(sr_imgs, y)
                        pixel_loss = self.pixel_weight * self.mse_loss(sr_imgs, y)
                        adversarial_loss = self.adv_weight * self.adv_loss(sr_discriminated, torch.ones_like(sr_discriminated))
                        if self.pixel_weight != 1:
                            g_loss = pixel_loss + vgg_loss + adversarial_loss
                        else:
                            g_loss = pixel_loss + adversarial_loss

                    if isBackward:
                        self.scaler.scale(g_loss).backward()
                        self.scaler.step(self.optimizerG)
                        self.scaler.update()

                    for d_parameter in self.modelD.parameters():
                        d_parameter.requires_grad = True

                    self.modelD.zero_grad(set_to_none=True)
                    #Update discriminator
                    with amp.autocast():
                        gt_output = self.modelD(y)
                        d_loss_gt = self.adv_loss(gt_output, real_label)

                    if isBackward:
                        self.scaler.scale(d_loss_gt).backward()

                    with amp.autocast():
                        sr_output = self.modelD(sr_imgs.detach().clone())
                        d_loss_sr = self.adv_loss(sr_output, fake_label)

                    if isBackward:
                        self.scaler.scale(d_loss_sr).backward()

                    d_loss = d_loss_gt + d_loss_sr
                    if isBackward:
                        # Update discriminator model weights
                        self.scaler.step(self.optimizerD)
                        self.scaler.update()

                else:
                    sr_imgs = self.modelG(x)
                    g_loss = self.l1_loss(sr_imgs, y)
                    self.optimizerG.zero_grad()
                    if isBackward:
                        g_loss.backward()
                    self.optimizerG.step()

                running_results['iterations'] += 1

                if not generatorOnly:
                    total_dLoss += d_loss.item()
                if not discriminatorOnly:
                    totalLoss += g_loss.item()
                    running_results['g_loss'] += g_loss.item()
                if discriminatorOnly:
                    running_results['d_loss'] += d_loss.item()
                    batchLoader.set_description(desc="[%d/20] Loss_D: %.4f " % (
                        self.epoch + 1,
                        running_results['d_loss'] / running_results['iterations']
                    ) + f"Learning rateD: {'%.1f' % (-math.log(self.optimizerD.param_groups[0]['lr'], 10))}")
                elif not generatorOnly:
                    running_results['d_loss'] += d_loss
                    batchLoader.set_description(desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f " % (
                        self.epoch + 1, self.configs['epochs'],
                        running_results['d_loss'] / running_results['iterations'],
                        running_results['g_loss'] / running_results['iterations']
                    ) + f"Learning rateG: {'%.1f' % (-math.log(self.learningRate, 10))} "
                                                     + f"Learning rateD: {'%.1f' % (-math.log(self.optimizerD.param_groups[0]['lr'], 10))}")
                else:
                    batchLoader.set_description(desc=f'[{self.epoch}/{self.configs["epochs"]}] '
                                                     f"Learning rate: {'%.1f' % (-math.log(self.learningRate, 10))} "
                                                     f'Loss_G: {"%.4f" % (running_results["g_loss"] / running_results["iterations"])}')

                # Additional metrics
                label = np.clip(torch.squeeze(y).cpu().detach().numpy(), 0, 1)
                label = ((label * 255) / np.max(label)).astype(np.uint8)
                output = np.clip(torch.squeeze(sr_imgs).cpu().detach().numpy(), 0, 1)
                output = ((output * 255) / np.max(output)).astype(np.uint8)
                psnr = cv2.PSNR(output, label)
                psnr_list.append(psnr)

                ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
                ssim_val = ssim(sr_imgs.cpu().detach(), y.cpu().detach())
                ssim_list.append(ssim_val.item())

        return totalLoss / len(batchLoader), total_dLoss / len(batchLoader), \
               sum(psnr_list) / len(batchLoader), sum(ssim_list) / len(batchLoader)

    def train(self, loader=None):
        title("Train")
        self.trainloader = loader or self.trainloader
        self.load()
        self.trainEpochs(self.startEpoch, self.configs["epochs"])

    def test(self):
        title("Test")
        self.loadBest()

    def saveObject(self, epoch):
        return {
            "epoch": epoch,
            "modelG": self.modelG.state_dict(),
            "schedulerG": self.schedulerG.state_dict(),
            "optimizerG": self.optimizerG.state_dict(),
            "modelD": self.modelD.state_dict(),
            "schedulerD": self.schedulerD.state_dict(),
            "optimizerD": self.optimizerD.state_dict(),
            "trainLosses": self.trainLosses,
            "validLosses": self.validLosses,
            "learningRates": self.learningRates,
            "trainPsnrs": self.trainPsnrs,
            "validPsnrs": self.validPsnrs,
            "trainSsims": self.trainSsims,
            "validSsims": self.validSsims
        }

    def getCheckpointFolder(self, generator_only=False):
        if generator_only:
            return PATHS.CHECKPOINTS / f"X{self.configs['scaleFactor']}" / (self.generatorOnlyName())
        return PATHS.CHECKPOINTS / f"X{self.configs['scaleFactor']}" / (self.getModelName())

    def getModelName(self):
        return f"{MODEL_NAME}-lr{self.configs['startLearningRate']}-generatorOnly{self.pretrain}-flip{self.configs['randomFlip']}-psize" \
               f"{self.configs['patchSize']}-pixelweight-{self.pixel_weight}"

    def generatorOnlyName(self):
        return f"ESRT-lr{self.configs['startLearningRate']}-flip{self.configs['randomFlip']}-psize" \
               f"{self.configs['patchSize']}"

    def save(self, fileName="", pretrained='false'):
        epoch = self.epoch
        fileName = fileName or f"epoch{epoch + 1}.pth"
        saveFolder = self.getCheckpointFolder()
        mkdir(saveFolder)
        fileName = saveFolder / fileName
        torch.save(self.saveObject(epoch), fileName)

    def load(self):
        saveFolder = self.getCheckpointFolder()
        startEpoch = self.configs["startEpoch"]

        startEpoch = getFinalEpoch(saveFolder) if startEpoch == -1 else startEpoch

        #For GAN load pretrained generator on start
        if not self.pretrain and startEpoch == 0:
            saveFolder = self.getCheckpointFolder(generator_only=True)
            startEpoch = getFinalEpoch(saveFolder)
            if startEpoch == 0:
                return  # if 0 no load (including can't find )
            self.startEpoch = startEpoch
            modelFile = getFile(saveFolder, f"epoch{startEpoch}.pth")
            self.loadParams(modelFile, only_generator=True)
            self.startEpoch = 0
        else:
            self.startEpoch = startEpoch
            if startEpoch == 0:
                return  # if 0 no load (including can't find )
            modelFile = getFile(saveFolder, f"epoch{startEpoch}.pth")

            self.loadParams(modelFile)

    def loadBest(self):
        modelFile = getFile(self.getCheckpointFolder(), "best*.pth")
        if modelFile:
            self.loadParams(modelFile)
        else:
            warn(f"best model not found under {self.getCheckpointFolder()}\nIs 'bestXXX.pth' exist?")
            self.load()

    def loadParams(self, fileP, only_generator=False):
        info(f"load model from {fileP.name}")
        saveObject = torch.load(fileP)

        if only_generator:
            self.modelG.load_state_dict(saveObject["model"])
        else:
            self.modelG.load_state_dict(saveObject["modelG"])

        if only_generator:
            self.optimizerG = optim.Adam(self.modelG.parameters(), lr=self.configs["startLearningRate"])
            self.schedulerG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=100, gamma=0.5)
        else:
            self.schedulerG.load_state_dict(saveObject["schedulerG"])
            self.optimizerG.load_state_dict(saveObject["optimizerG"])
            if not self.pretrain:
                self.modelD.load_state_dict(saveObject["modelD"])
                self.schedulerD.load_state_dict(saveObject["schedulerD"])
                self.optimizerD.load_state_dict(saveObject["optimizerD"])
            self.validLosses = saveObject["validLosses"]
            self.trainLosses = saveObject["trainLosses"]
            self.trainPsnrs = saveObject["trainPsnrs"]
            self.validPsnrs = saveObject["validPsnrs"]
            self.trainSsims = saveObject["trainSsims"]
            self.validSsims = saveObject["validSsims"]
            self.learningRates = saveObject["learningRates"]
            self.bestValidLoss = max([*self.validLosses, 0])

    def initParams(self):
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.modelG = ESRT(upscale=self.configs["scaleFactor"])
        self.modelG = self.modelG.to(self.device)
        self.optimizerG = optim.Adam(self.modelG.parameters(), lr=self.configs["startLearningRate"], betas=(0.9, 0.999))
        self.schedulerG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=50, gamma=0.1)

        self.modelD = Discriminator()
        self.modelD = self.modelD.to(self.device)
        self.optimizerD = optim.Adam(self.modelD.parameters(), lr=self.configs["startLearningRate"], betas=(0.9, 0.999))
        self.schedulerD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=50, gamma=0.1)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.vgg_loss = VGGLoss()
        self.content_loss = nn.MSELoss().to(self.device)
        self.adv_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.scaler = amp.GradScaler()
        self.pixel_weight = self.configs["pixelWeight"]
        self.adv_weight = self.configs["advWeight"]
        self.pretrain = self.configs["pretrainG"]
        self.trainLosses = []
        self.trainPsnrs = []
        self.validLosses = []
        self.trainSsims = []
        self.validLosses = []
        self.validPsnrs = []
        self.validSsims = []
        self.learningRates = []
        self.bestValidLoss = float("inf")
        self.batchSize = self.configs["batchSize"]
        self.trainDatasetPath = PATHS.DATASETS / self.configs["datasetPath"]
        self.patchSize = self.configs["patchSize"]
        mean_imagenet = [0.485, 0.456, 0.406]
        std_imagenet = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean_imagenet, std_imagenet)
        self.trainDataset = DFO2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            crop_size=self.patchSize,
            is_training=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        self.validDataset = DFO2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            crop_size=self.patchSize,
            is_training=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        self.trainloader = DataLoader(
            self.trainDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"], persistent_workers=True)
        self.validloader = DataLoader(
            self.validDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"], persistent_workers=True)
