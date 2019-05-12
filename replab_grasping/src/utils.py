import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
import cv2
from scipy.interpolate import NearestNDInterpolator

from replab_core.config import *


def Rx(x):
    return np.matrix([[1, 0, 0],
                      [0, np.cos(x), -np.sin(x)],
                      [0, np.sin(x), np.cos(x)]])


def Ry(y):
    return np.matrix([[np.cos(y), 0, np.sin(y)],
                      [0, 1, 0],
                      [-np.sin(y), 0, np.cos(y)]])


def Rz(z):
    return np.matrix([[np.cos(z), -np.sin(z), 0],
                      [np.sin(z), np.cos(z), 0],
                      [0, 0, 1]])


def Rt(rvec, tvec):
    R = np.dot(Rz(rvec[2]), np.dot(Ry(rvec[1]), Rx(rvec[0])))
    Rt = np.matrix([[R[0, 0], R[0, 1], R[0, 2], tvec[0]],
                    [R[1, 0], R[1, 1], R[1, 2], tvec[1]],
                    [R[2, 0], R[2, 1], R[2, 2], tvec[2]],
                    [0.0, 0.0, 0.0, 1.0]])
    return Rt


def apply_Rt(x, Rt):
    x = np.concatenate([x, [1]])
    return np.dot(Rt, x)[:, :3]


def inside_polygon(point, arena, height_constraints=None):
    # Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    x, y, z = point

    n = len(arena)
    inside = False
    p1x, p1y = arena[0]
    for i in range(1, n + 1):
        p2x, p2y = arena[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    if height_constraints:
        lower, upper = min(height_constraints), max(height_constraints)
        return inside and z >= lower and z <= upper
    else:
        return inside


def crop_image(img, center, radius=48):
    y, x = center[0] + radius, center[1] + radius
    img = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), 'constant')
    cropped = img[x - radius:x + radius, y - radius:y + radius, :]
    return cropped


def rotate_image(img, angle):
    rotated = rotate(img, angle, reshape=True)
    return rotated


def make_resize(x, y):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((x, y)),
        transforms.ToTensor()])


def make_resize_rgb(x, y):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((x, y)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])


def get_probability(s0, s1):
    p0, p1 = np.exp(s0), np.exp(s1)
    return p1 / (p0 + p1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_blobs(pc):
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                n_jobs=-1).fit(pc)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    cluster_centers = []

    for cluster in set(db.labels_):
        if cluster != -1:
            running_sum = np.array([0.0, 0.0, 0.0])
            counter = 0

            for i in range(pc.shape[0]):
                if labels[i] == cluster:
                    running_sum += pc[i]
                    counter += 1

            center = running_sum / counter
            center[2] -= Z_OFFSET

            if center[2] > Z_MIN:
                center[2] = Z_MIN
                cluster_centers.append(center)
            elif center[2] < PRELIFT_HEIGHT:
                for i, label in enumerate(labels):
                    if label == cluster:
                        labels[i] = -1
                continue
            else:
                cluster_centers.append(center)

    print('Blobs detected: %d' % len(cluster_centers))

    return cluster_centers, labels


def rectify_depth(depth, interpolate=True):
    rectified = cv2.rgbd.registerDepth(
        DEPTH_K, RGB_K, None, DEPTH_TO_RGB_RT, depth, (640, 480), False)
    rectified = np.nan_to_num(rectified)
    if interpolate:
        nndi = NearestNDInterpolator(np.argwhere(
            rectified != 0.), rectified[rectified != 0.])
        missing_indices = np.argwhere(rectified == 0.)
        interpolated = nndi(missing_indices)
        rectified[missing_indices[:, 0], missing_indices[:, 1]] = interpolated
    return rectified
