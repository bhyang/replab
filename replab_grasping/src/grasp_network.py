import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet

import numpy as np
import functools


class FullImageNet(nn.Module):

    def __init__(self, ignore_grasp=False, filters=32):
        super(FullImageNet, self).__init__()
        self.ignore_grasp = ignore_grasp
        self.filters = filters

        self.rgb_features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=6, padding=2, stride=2),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        self.d_features = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=6, padding=2, stride=2),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        self.grasp_features = nn.Sequential(
            nn.Linear(4, filters),
            nn.ReLU(inplace=True),
            nn.Linear(filters, filters)
        )

        self.conv_features = nn.Sequential(
            nn.Conv2d(2 * filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(filters * 36, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        rgb, depth, grasp = x
        rgb = self.rgb_features(rgb)
        depth = self.d_features(depth)
        img = torch.cat([rgb, depth], dim=1)

        if self.ignore_grasp:
            z = img
        else:
            grasp = self.grasp_features(grasp)
            grasp = grasp.reshape(grasp.size(0), self.filters, 1, 1)
            grasp = grasp.repeat(1, 2, 12, 12)
            z = img + grasp

        z = self.conv_features(z)
        z = z.view(z.size(0), 36 * self.filters)
        z = self.classifier(z)
        return z


class PintoGuptaNet(nn.Module):

    def __init__(self, depth=True, binned_output=True):
        super(PintoGuptaNet, self).__init__()
        self.depth = depth
        self.rgb_features = alexnet(pretrained=True).features

        if self.depth:
            self.d_features = alexnet(pretrained=True).features

        self.binned_output = binned_output

        self.fc = nn.Sequential(
            nn.Linear((2 if self.depth else 1) * 9216, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(.5),
            nn.Linear(256, 18 if binned_output else 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        rgb, depth, grasp = x
        rgb = self.rgb_features(rgb)
        rgb = rgb.view(rgb.size(0), 9216)

        if self.depth:
            depth = torch.cat([depth, depth, depth], dim=1)
            depth = self.d_features(depth)
            depth = depth.view(depth.size(0), 9216)
            img = torch.cat([rgb, depth], dim=1)
            out = self.fc(img)
        else:
            out = self.fc(rgb)

        if self.binned_output:
            theta = grasp[:, 3]
            bin_num = (theta / (np.pi / 18)).trunc()
            bin_num = torch.stack([torch.arange(rgb.shape[0]).type(torch.cuda.LongTensor),
                                   bin_num.type(torch.cuda.LongTensor)])
            return out[bin_num[0], bin_num[1]]
        else:
            return out
