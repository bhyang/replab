#!/usr/bin/env python

import numpy as np
from matplotlib.patches import Circle
import h5py

import rospy
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import (
    Image,
    PointCloud2,
    CameraInfo,
    JointState
)
from std_msgs.msg import (
    UInt16,
    Int64,
    Float32,
    Header
)
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError

from moveit_commander import PlanningSceneInterface
from moveit_commander.exception import MoveItCommanderException

from itertools import product
import time
import traceback
import argparse

from sklearn.neighbors import KDTree
from scipy.ndimage import rotate

from replab_core.controller import WidowX
from replab_core.config import *
from replab_core.utils import *


class Executor:

    def __init__(self, scan=False, datapath='', save=False):
        # WidowX controller interface
        self.widowx = WidowX()

        # Register subscribers
        self.img_subscriber = rospy.Subscriber(
            RGB_IMAGE_TOPIC, Image, self.update_rgb)
        self.depth_subscriber = rospy.Subscriber(
            DEPTH_IMAGE_TOPIC, Image, self.update_depth)
        self.pc_subscriber = rospy.Subscriber(
            POINTCLOUD_TOPIC, PointCloud2, self.update_pc)
        self.caminfo_subscriber = rospy.Subscriber(
            DEPTH_CAMERA_INFO_TOPIC, CameraInfo, self.save_cinfo)

        self.datapath = datapath
        self.save = save

        # For ROS/cv2 conversion
        self.bridge = CvBridge()
        self.transform = lambda x: x

        # Tracking misses for calling reset routine
        self.running_misses = 0

        self.evaluation_data = []

        # Store latest RGB-D
        self.rgb = np.zeros((480, 640, 3))
        self.depth = np.zeros((480, 640, 1))
        self.pc = np.zeros((1, 3))

        self.camera_info = CameraInfo()

        if scan:
            self.base_pc = np.zeros((1, 3)) + 5000.0  # adds dummy point
        else:
            self.base_pc = np.load(PC_BASE_PATH)

        self.cm = CALIBRATION_MATRIX

        self.inv_cm = np.linalg.inv(self.cm)

        self.sample = {}

        self.kdtree = KDTree(self.base_pc)

        rospy.sleep(2)

        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(self.camera_info)

    def update_rgb(self, data):
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        self.rgb = cv_image

    def update_depth(self, data):
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        self.depth = cv_image

    def update_pc(self, data):
        self.pc = pc2.read_points(data, skip_nans=True)

    def save_cinfo(self, data):
        self.camera_info = data

    def get_rgbd(self):
        old_depth = self.depth.astype(np.float) / 10000.
        depth = rectify_depth(old_depth)
        depth = np.reshape(depth, (480, 640, 1))
        return np.concatenate([self.rgb, depth], axis=2)

    def get_pose(self):
        pose = self.widowx.get_current_pose().pose
        pose_list = [pose.position.x,
                     pose.position.y,
                     pose.position.z,
                     pose.orientation.w,
                     pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z]
        return pose_list

    def get_pc(self):
        pc = list(self.pc)
        if len(pc) == 0:
            return []
        pc = np.array(pc)[:, :3]
        np.random.shuffle(pc)
        if pc.shape[0] > 5000:
            pc = pc[:5000]

        def transform_pc(srcpc, tf_matrix):
            ones = np.ones((srcpc.shape[0], 1))
            srcpc = np.append(srcpc, ones, axis=1)
            out = np.dot(tf_matrix, srcpc.T)[:3]
            return out.T

        pc = transform_pc(pc, self.cm)

        self.sample['full_pc'] = pc

        dist, _ = self.kdtree.query(pc, k=1)
        pc = [p for i, p in enumerate(pc) if inside_polygon(
            p, PC_BOUNDS, HEIGHT_BOUNDS) and dist[i] > .003]
        return np.reshape(pc, (len(pc), 3))

    def scan_base(self, scans=100):
        def haul(pc):
            try:
                pc = self.get_pc()
                print('# of new base points: %d' % len(pc))
                self.base_pc = np.concatenate([self.base_pc, pc], axis=0)
                np.save(PC_BASE_PATH, self.base_pc)
                self.kdtree = KDTree(self.base_pc)
            except ValueError as ve:
                traceback.print_exc(ve)
                print('No pointcloud detected')

        for i in range(scans):
            print('Scan %d' % i)
            haul(self.pc)
            rospy.sleep(1)

    def evaluate_grasp(self, manual=False):
        success, closure = self.widowx.eval_grasp(manual=manual)
        return success

    def execute_grasp(self, grasp, manual_label=False):
        try:
            x, y, z, theta = grasp

            print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f)'
                  % (x, y, z, theta))

            self.sample['attempted_parameters'] = grasp

            self.sample['before_img'] = self.get_rgbd()
            self.before = self.sample['before_img']

            assert inside_polygon(
                (x, y, z), END_EFFECTOR_BOUNDS), 'Grasp not in bounds'

            assert self.widowx.orient_to_pregrasp(
                x, y), 'Failed to orient to target'

            assert self.widowx.move_to_grasp(x, y, PRELIFT_HEIGHT, theta), \
                'Failed to reach pre-lift pose'

            assert self.widowx.move_to_grasp(
                x, y, z, theta), 'Failed to execute grasp'

            self.sample['pose'] = self.get_pose()
            self.sample['joints'] = self.widowx.get_joint_values()

            self.widowx.close_gripper()

            reached = self.widowx.move_to_vertical(PRELIFT_HEIGHT)

            assert self.widowx.move_to_drop(), 'Failed to move to drop'

            rospy.sleep(2)

            self.sample['after_img'] = self.get_rgbd()
            self.after = self.sample['after_img']

            success = self.evaluate_grasp(manual=manual_label)
            self.sample['gripper_closure'] = self.widowx.eval_grasp()[1]

            return success, 0

        except Exception as e:
            print('Error executing grasp -- returning...')
            traceback.print_exc(e)
            return False, 1

    def calculate_crop(self, grasp):
        grasp = np.concatenate([grasp, [1.]], axis=0)
        transformedPoint = np.dot(self.inv_cm, grasp)
        predicted = self.camera.project3dToPixel(transformedPoint)
        return int(predicted[0]), int(predicted[1])

    def save_sample(self, i=0):
        self.sample['timestamp'] = time.ctime()

        self.sample['D'] = self.camera_info.D
        self.sample['K'] = self.camera_info.K
        self.sample['R'] = self.camera_info.R
        self.sample['P'] = self.camera_info.P

        graspPoint = np.array([self.sample['pose'][0], self.sample['pose'][1],
                               self.sample['pose'][2]])

        predicted = self.calculate_crop(graspPoint)

        self.sample['pixel_point'] = predicted

        before = self.sample['before_img'][:, :, :3].astype(np.uint8)

        with h5py.File(self.datapath + str(i) + '.hdf5', 'w') as file:
            for key in self.sample:
                file[key] = self.sample[key]

        self.sample = {}

        print('Saved to %s' % self.datapath + str(i) + '.hdf5')
