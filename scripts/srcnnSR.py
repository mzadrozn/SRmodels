import math
import cv2
import torch
from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.SRCNN.models import SRCNN
from datasets.prepare_training_data import DFO2KDataset
from .configParser import ConfigParser
from .utils import *
from scripts.appSR import AppSR
import numpy as np
MODEL_NAME = "SRCNN"


class SrcnnSR(AppSR):
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
        for epoch in range(start, end):
            self.epoch = epoch
            trainLoss, mean_psnr, mean_ssim = self.epochAction("train", self.trainloader)
            self.trainLosses.append(trainLoss.item())
            self.trainPsnrs.append(mean_psnr)
            self.trainSsims.append(mean_ssim)
            validLoss, mean_psnr, mean_ssim = self.epochAction("valid", self.validloader)
            self.validLosses.append(validLoss.item())
            self.learningRates.append(self.learningRate)
            self.validSsims.append(mean_ssim)
            self.validPsnrs.append(mean_psnr)
            self.scheduler.step()

            if validLoss < self.bestValidLoss:
                self.bestValidLoss = validLoss
                [best.unlink() for best in getFiles(self.getCheckpointFolder(), "best*.pth")]  # remove last best pth
                self.save(f"bestEpoch{epoch + 1}.pth")
                info(f"save best model, valid loss {round(validLoss.item(), 3)}")

            if (epoch + 1) % self.configs["saveEvery"] == 0:
                self.save()

    @property
    def learningRate(self):
        return self.optimizer.param_groups[0]['lr']

    def modelForward(self, x, y):
        device = self.device
        x, y = map(lambda t: rearrange(t.to(device), 'b p c h w -> (b p) c h w'), (x, y))
        out = self.model(x)
        loss = self.criterion(out, y)
        return x, y, out, loss

    def epochAction(self, action, loader):
        totalLoss, totalCorrect, totalLen = 0, 0, 0
        psnr_list, ssim_list = [], []
        isBackward = True if action == "train" else False
        GradSelection = Grad if isBackward else torch.no_grad
        batchLoader = tqdm(loader)
        if isBackward:
            self.model.train()
        else:
            self.model.eval()
        with GradSelection():
            for x, y in batchLoader:
                self.optimizer.zero_grad()
                device = self.device
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)
                out = self.model(x)
                loss = self.criterion(out, y)

                totalLoss += loss
                totalLen += 1
                if isBackward:
                    loss.backward()
                    self.optimizer.step()
                epochProgress = f"{self.epoch + 1}/{self.configs['epochs']}" if action != "test" else "1/1"

                # Additional metrics
                label = np.clip(torch.squeeze(y).cpu().detach().numpy(), 0, 1) * 255
                label = label.astype(np.uint8)
                output = np.clip(torch.squeeze(out).cpu().detach().numpy(), 0, 1) * 255
                output = output.astype(np.uint8)
                psnr = cv2.PSNR(output, label)
                psnr_list.append(psnr)

                ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
                ssim_val = ssim(out.cpu().detach(), y.cpu().detach())
                ssim_list.append(ssim_val.item())

                batchLoader.set_description(
                    desc=f"{action} [{epochProgress}] -lglr {'%.1f' % (-math.log(self.learningRate, 10))} 🕐loss "
                         f"{'%.4f' % (loss)} ➗loss {'%.4f' % (totalLoss / totalLen)} mean psnr:"
                         f"{sum(psnr_list) / totalLen}")

        return totalLoss / len(batchLoader),  sum(psnr_list) / len(batchLoader), sum(ssim_list) / len(batchLoader)

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
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainLosses": self.trainLosses,
            "validLosses": self.validLosses,
            "learningRates": self.learningRates,
            "trainPsnrs": self.trainPsnrs,
            "validPsnrs": self.validPsnrs,
            "trainSsims": self.trainSsims,
            "validSsims": self.validSsims
        }

    def getCheckpointFolder(self):
        return PATHS.CHECKPOINTS / f"X{self.configs['scaleFactor']}" / self.getModelName()

    def getModelName(self):
        return f"SRCNN-lr{self.configs['startLearningRate']}-flip{self.configs['randomFlip']}-psize{self.patchSize}"

    def save(self, fileName=""):
        epoch = self.epoch
        fileName = fileName or f"epoch{epoch + 1}.pth"
        saveFolder = self.getCheckpointFolder()
        mkdir(saveFolder)
        fileName = saveFolder / fileName
        torch.save(self.saveObject(epoch), fileName)

    def load(self):
        saveFolder = self.getCheckpointFolder()
        startEpoch = self.configs["startEpoch"]

        startEpoch = getFinalEpoch(saveFolder) if startEpoch == -1 else startEpoch  # get real last epoch if -1
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

    def loadParams(self, fileP):
        info(f"load model from {fileP.name}")
        saveObject = torch.load(fileP)
        self.model.load_state_dict(saveObject["model"])
        self.scheduler.load_state_dict(saveObject["scheduler"])
        self.optimizer.load_state_dict(saveObject["optimizer"])
        self.validLosses = saveObject["validLosses"]
        self.trainLosses = saveObject["trainLosses"]
        self.trainPsnrs = saveObject["trainPsnrs"]
        self.validPsnrs = saveObject["validPsnrs"]
        self.trainSsims = saveObject["trainSsims"]
        self.validSsims = saveObject["validSsims"]
        self.learningRates = saveObject["learningRates"]
        self.bestValidLoss = min(self.validLosses)

    def initParams(self):
        self.criterion = torch.nn.L1Loss()
        self.model = SRCNN(num_chanels=3)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["startLearningRate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.trainLosses = []
        self.trainPsnrs = []
        self.trainSsims = []
        self.validLosses = []
        self.validPsnrs = []
        self.validSsims = []
        self.learningRates = []
        self.bestValidLoss = float("inf")
        self.batchSize = self.configs["batchSize"]
        self.patchSize = self.configs['patchSize']
        self.trainDatasetPath = PATHS.DATASETS / self.configs["datasetPath"]

        self.trainDataset = DFO2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            crop_size=self.patchSize,
            is_training=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            interpolate=True
        )
        self.validDataset = DFO2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            crop_size=self.patchSize,
            is_training=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            interpolate=True
        )
        self.trainloader = DataLoader(
            self.trainDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"], persistent_workers=True)
        self.validloader = DataLoader(
            self.validDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"], persistent_workers=True)


if __name__ == '__main__':
    a = SrcnnSR("train")


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))