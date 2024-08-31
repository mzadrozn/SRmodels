import torch.nn as nn
from torchvision.models import vgg19
import models.SRGAN.config as config

FEATURE_MAP_SCALE = 12.5


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input) / FEATURE_MAP_SCALE
        vgg_target_features = self.vgg(target) / FEATURE_MAP_SCALE
        return self.loss(vgg_input_features, vgg_target_features)
