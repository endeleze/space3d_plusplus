from __future__ import print_function, division

from torchvision import models

import torch.nn as nn

import torch

import numpy as np


class I2P(nn.Module):

    def __init__(self, model_type=None):

        super(I2P, self).__init__()

        if model_type == "vgg16":
            self.model = models.vgg16(pretrained=False)

            self.model.features[0] = torch.nn.Conv2d(in_channels=12, out_channels=64,

                                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            nn.init.kaiming_normal_(self.model.features[0].weight, mode='fan_out', nonlinearity='relu')

            nn.init.constant_(self.model.features[0].bias, 0)

            self.model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4),
            )

            nn.init.normal_(self.model.classifier[0].weight, 0, 0.01)

            nn.init.constant_(self.model.classifier[0].bias, 0)

            nn.init.normal_(self.model.classifier[2].weight, 0, 0.01)

            nn.init.constant_(self.model.classifier[2].bias, 0)

            nn.init.normal_(self.model.classifier[4].weight, 0, 0.01)

            nn.init.constant_(self.model.classifier[4].bias, 0)

        if model_type == "resnet50":
            self.model = models.resnet50(pretrained=False)

            self.model.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=64,

                                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

            num_ftrs = self.model.fc.in_features

            self.model.fc = torch.nn.Linear(num_ftrs, 4)

            nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_out', nonlinearity='relu')

            nn.init.constant_(self.model.fc.bias, 0)


    def forward(self, x):

        y = self.model(x)

        return y