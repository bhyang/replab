import numpy as np
from scipy.linalg import eigh

from grasp_network import FullImageNet, PintoGuptaNet
from replab_core.config import *
from replab_core.utils import *

import traceback

import torch.nn as nn
import torchvision.transforms as transforms


class Policy:

    def plan_grasp(self, rgbd, pc):
        '''
        Parameters:
            rgbd: the RGB-D image (480 x 640 x 4 numpy array)
            pc: the filtered point cloud (N x 3 numpy array)
        Returns a list of tuples of grasps and associated confidences
            [([x,y,z,theta], confidence), ...]
        '''


class Custom(Policy):
    '''
    User-implemented policy that is called as "custom"
    '''

    def __init__(self):
        raise NotImplementedError

    def plan_grasp(self, rgbd, pc):
        raise NotImplementedError


class DataCollection(Policy):
    '''
    Grasps near the center of the detected object with random wrist orientation. Used for data collection
    '''

    def __init__(self, noise=True):
        self.noise = noise

    def plan_grasp(self, rgbd, pc):
        blobs, _ = compute_blobs(pc)
        index = np.random.randint(0, len(blobs))
        center = blobs[index]

        if center[2] < Z_MIN:
            new_z = np.random.uniform(center[2], Z_MIN)
            center[2] = new_z

        if self.noise:
            center = np.append(center, [0.])
            noise = np.random.uniform([-XY_NOISE, -XY_NOISE, .0, -1.57],
                                      [XY_NOISE, XY_NOISE, .0, 1.57])
            return [(center + noise, 1.)]
        else:
            center = np.append(center, np.random.uniform(-1.57, 1.57, (1)))
            return [(center, 1.)]


class PrincipalAxis(Policy):
    '''
    Grasps the center of the object with theta perpendicular to the principal axis
    '''

    def plan_grasp(self, rgbd, pc):
        blobs, labels = compute_blobs(pc)
        thetas = []
        confidences = []

        for i in range(len(set(labels)) - 1):
            x = pc[labels == i][:, :2]
            x = x - np.mean(x, axis=0)
            xx = np.dot(x.T, x)
            w, v = eigh(xx)
            confidences.append(np.abs(w[-1]) / np.abs(w[0]))

            eigv = v[-1]
            center = np.mean(x, axis=0)
            thetas.append(np.arctan2(eigv[1], eigv[0]) % np.pi)

        return [((blobs[i][0], blobs[i][1], Z_MIN, thetas[i]), confidence)
                for i, confidence in enumerate(confidences)]


class Pinto2016(Policy):
    '''
    Implementation of Pinto 2016
    Details found here: https://arxiv.org/pdf/1509.06825.pdf
    '''

    def __init__(self, model_path=None, heightmaps=False):
        self.net = PintoGuptaNet(depth=False, binned_output=True).cuda()
        self.net = nn.DataParallel(self.net).cuda()
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.resize = make_resize_rgb(227, 227)

        self.K = DEPTH_K

        self.cm = CALIBRATION_MATRIX

        self.inv_cm = np.linalg.inv(self.cm)

    def calculate_crops(self, grasps):
        grasps = np.concatenate([grasps, np.ones((len(grasps), 1))], axis=1)
        camera_points = np.dot(self.inv_cm, grasps.T)[:3]
        camera_points = (camera_points.T + np.array([0.026, 0.001, 0.004])).T
        pixel_points = np.dot(self.K, camera_points / camera_points[2:])[:2].T
        return pixel_points.astype(int)

    def plan_grasp(self, rgbd, pc, num_grasps=256, batch_size=128):
        rgb = rgbd[:, :, :3].astype(np.uint8)
        _, labels = compute_blobs(pc)

        blobs = []
        for label in set(labels):
            if label == -1:
                continue
            blob_points = pc[labels == label]
            index = np.random.randint(0, len(blob_points))
            blobs.append(blob_points[index])

        all_grasps = []
        all_probabilities = []
        all_crops = []

        for blob in blobs:
            blob = np.concatenate([blob, [0.]], axis=0)
            blob[2] = Z_MIN

            candidates = []
            probabilities = []
            cropss = []

            for i in range(num_grasps // batch_size):
                noise = np.random.uniform([-XY_NOISE, -XY_NOISE, -.02, -1.57], [XY_NOISE, XY_NOISE, 0.0, 1.57],
                                          (batch_size, 4))
                grasps = noise + blob
                candidates.append(grasps)

                crops = self.calculate_crops(grasps[:, :3])

                cropped = []
                for crop in crops:
                    img = crop_image(rgb, crop, 48)
                    cropped.append(self.resize(img))

                cropss.append(crops)

                rgbs = torch.stack(cropped).cuda()
                grasps = torch.tensor(grasps, dtype=torch.float).cuda()

                output = self.net.forward((rgbs, None, grasps))

                probabilities.extend([sigmoid(k)
                                      for k in output.detach().cpu().numpy()])
            candidates = np.concatenate(candidates, axis=0)
            best_indices = np.argsort(probabilities)[-5:]
            best_index = np.random.choice(best_indices)
            cropss = np.concatenate(cropss, axis=0)

            all_crops.append(cropss[best_index])
            all_grasps.append(candidates[best_index])
            all_probabilities.append(probabilities[best_index])

        all_grasps = np.array(all_grasps)

        return [(grasp, all_probabilities[i]) for i, grasp in enumerate(all_grasps)]


class FullImage(Policy):
    '''
    Uses a model trained to predict grasp success from the full scene image and the grasp configuration to plan grasps
    '''

    def __init__(self, model_path=None):
        self.net = FullImageNet().cuda()
        self.net = nn.DataParallel(self.net).cuda()
        self.net.load_state_dict(torch.load(model_path), strict=False)
        self.net.eval()
        self.resize = make_resize(227, 227)

    def plan_grasp(self, rgbd, pc, num_grasps=2048, batch_size=64):
        rgb, depth = rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3]

        depth = depth.astype(np.float)
        depth /= (MAX_DEPTH / 255.0)
        depth = depth.astype(np.uint8)
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))

        rgb, depth = self.resize(rgb), self.resize(depth)

        blobs, labels = compute_blobs(pc)

        all_grasps = []
        all_probabilities = []

        for blob in blobs:
            blob = np.concatenate([blob, [0.]], axis=0)
            blob[2] = Z_MIN

            candidates = []
            probabilities = []

            for i in range(num_grasps // batch_size):
                noise = np.random.uniform([-.02, -.02, -.02, 0], [.02, .02, 0.0, 3.14 / 2],
                                          (batch_size, 4))
                grasps = noise + blob
                candidates.append(grasps)

                grasps = torch.tensor(grasps, dtype=torch.float).cuda()

                rgb_batch, depth_batch = rgb.repeat(batch_size, 1, 1, 1).cuda(
                ), depth.repeat(batch_size, 1, 1, 1).cuda()

                output = self.net.forward((rgb_batch, depth_batch, grasps))

                probabilities.extend([get_probability(i, j)
                                      for i, j in output.detach().cpu().numpy()])

            candidates = np.concatenate(candidates, axis=0)
            best_index = np.argmax(probabilities)

            all_grasps.append(candidates[best_index])
            all_probabilities.append(probabilities[best_index])

        all_grasps = np.array(all_grasps)

        return [(grasp, all_probabilities[i]) for i, grasp in enumerate(all_grasps)]
