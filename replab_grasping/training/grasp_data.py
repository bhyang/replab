from __future__ import division

import numpy as np
import os
import h5py
import math

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from scipy.ndimage import rotate


def make_resize(x, y):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((x, y)),
        transforms.ToTensor()
    ])


def make_resize_jitter(x, y, train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((x, y)),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((x, y)),
            transforms.ToTensor()
        ])

GRAYSCALE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])
ADD_NOISE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    transforms.ToTensor()
])
NORMALIZE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
MAX_DEPTH = 700.0


def crop_image(img, center, radius=64):
    y, x = center[0] + radius, center[1] + radius
    img = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), 'constant')
    cropped = img[x - radius:x + radius, y - radius:y + radius, :]
    return cropped


def rotate_image(img, angle):
    rotated = rotate(img, angle, reshape=True)
    return rotated


def rotate_xy(xy, angle, size=(640, 480)):
    ox, oy = size[0] / 2, size[1] / 2
    px, py = xy

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    qx, qy = int(qx), int(qy)
    return np.array((qx, qy))


class GraspDetectionData(Dataset):

    def __init__(self, path='', valid_mask=[], train=True):
        self.path = path

        gripper_widths = np.load(path + 'widths.npy')
        successes = np.load(path + 'successes.npy')

        total_samples = successes.shape[0]
        self.num_samples = int(np.sum(valid_mask)) if len(
            valid_mask) > 0 else total_samples

        self.before_images = []
        self.after_images = []
        self.widths = []
        self.successes = []

        for i in range(total_samples):
            if len(valid_mask) == 0 or valid_mask[i]:
                self.before_images.append(path + 'before/' + str(i) + '.npy')
                self.after_images.append(path + 'after/' + str(i) + '.npy')
                self.widths.append(gripper_widths[i])
                self.successes.append(successes[i])
            self.train = train
        self.resize_rgb = make_resize_jitter(64, 64, self.train)
        self.resize_depth = make_resize(64, 64)

    def __getitem__(self, index):
        before = np.load(self.before_images[index])
        after = np.load(self.after_images[index])
        width = torch.tensor([self.widths[index]], dtype=torch.float)
        success = int(self.successes[index])

        rgb_list, depth_list = [], []

        for img in (before, after):
            rgb, depth = img[:240, 80:320, :3], img[:240, 80:320, 3]
            rgb = rgb.astype(np.uint8)

            depth = depth.astype(np.float)
            depth /= (MAX_DEPTH / 255.0)
            depth = depth.astype(np.uint8)
            depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))

            rgb_list.append(rgb)
            depth_list.append(depth)

        rgb, depth = rgb_list[1], depth_list[1]

        rgb, depth = self.resize_rgb(rgb), self.resize_depth(depth)
        return (rgb, depth, width), success

    def __len__(self):
        return self.num_samples


class StandardData(Dataset):

    def __init__(self, path='', valid_mask=[], random_rotate=False, train=True):
        self.path = path
        self.random_rotate = random_rotate

        grasps = np.load(path + 'grasps.npy')
        successes = np.load(path + 'successes.npy')

        total_samples = grasps.shape[0]
        self.num_samples = int(np.sum(valid_mask)) if len(
            valid_mask) > 0 else total_samples
        self.rgbd = []
        self.grasps = []
        self.successes = []

        for i in range(total_samples):
            if len(valid_mask) == 0 or valid_mask[i]:
                self.rgbd.append(path + 'before/' + str(i) + '.npy')
                self.grasps.append(grasps[i])
                self.successes.append(successes[i])

        self.resize_rgb = make_resize_jitter(227, 227, train)
        self.resize_depth = make_resize(227, 227)

    def __getitem__(self, index):
        img = np.load(self.rgbd[index])
        rgb, depth = img[:, :, :3], img[:, :, 3]
        grasp = torch.tensor(self.grasps[index], dtype=torch.float)
        success = int(self.successes[index])

        rgb = rgb.astype(np.uint8)

        depth = depth.astype(np.float)
        depth /= (MAX_DEPTH / 255.0)
        depth = depth.astype(np.uint8)
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))

        rgb, depth = self.resize_rgb(rgb), self.resize_depth(depth)
        return (rgb, depth, grasp), success

    def __len__(self):
        return self.num_samples


class CroppedData(Dataset):

    def __init__(self, path='', valid_mask=[], crop_radius=48, train=True):
        self.path = path
        self.crop_radius = crop_radius
        self.extended_radius = int(self.crop_radius * np.sqrt(2))

        grasps = np.load(path + 'grasps.npy')
        successes = np.load(path + 'successes.npy')

        crops = np.load(path + 'crops.npy')
        angles = np.load(path + 'angles.npy')

        if isinstance(valid_mask, type(None)):
            valid_mask = np.ones(grasps.shape[0]).astype(bool)
        else:
            valid_mask = valid_mask.astype(bool)

        valid_crops = (crops[:, 0] >= 111) & (crops[:, 0] < 461) & (crops[:, 1] >= 107) \
            & (crops[:, 1] < 560)

        valid_mask = valid_mask & valid_crops

        total_samples = grasps.shape[0]
        self.num_samples = int(np.sum(valid_mask)) if len(
            valid_mask) > 0 else total_samples

        self.rgbd = []
        self.grasps = []
        self.successes = []
        self.crops = []
        self.angles = []

        for i in range(total_samples):
            if len(valid_mask) == 0 or valid_mask[i]:
                self.rgbd.append(path + 'before/' + str(i) + '.npy')
                self.grasps.append(grasps[i])
                self.successes.append(successes[i])
                self.crops.append(crops[i])
                self.angles.append(angles[i])

        self.resize_rgb = make_resize_jitter(227, 227, train)
        self.resize_depth = make_resize(227, 227)

    def __getitem__(self, index):
        img = np.load(self.rgbd[index])
        rgb, depth = img[:, :, :3], img[:, :, 3]
        grasp = torch.tensor(self.grasps[index], dtype=torch.float)
        success = int(self.successes[index])
        crop = self.crops[index]
        angle = self.angles[index]

        rgb = rgb.astype(np.uint8)

        depth = depth.astype(np.float)
        depth /= (MAX_DEPTH / 255.0)
        depth = depth.astype(np.uint8)
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))

        angle = angle % np.pi
        rgb, depth = crop_image(rgb, crop, self.crop_radius), crop_image(
            depth, crop, self.crop_radius)
        grasp[3] = angle

        rgb, depth = self.resize_rgb(rgb), self.resize_depth(depth)

        return (rgb, depth, grasp), success

    def __len__(self):
        return self.num_samples
