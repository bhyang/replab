#!/usr/bin/env python

# @package click_window
#    This module uses OpenCV's HighGUI platform to import a camera stream and allow a
#    person to click on an arbitrary pixel at arbitrary levels of zoom. It outputs
#    a message containing the pixel value (at zoom 100%) and camera_info, to the
#    topic specified by "outputName"

import roslib
import sys
import rospy
import math
import tf
from tf.msg import tfMessage
import cv2
from std_msgs.msg import String
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrv
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from matplotlib.pylab import cm

import numpy as np
import matplotlib.pyplot as plt

# ClickWindow documentation
#
#    A class which, when instantiated, creates a clickable window from a camera stream
#    Note that this class does not automatically persist: it must have its .listen() function
#    called periodically. This may change in future releases.


class ClickWindow:

    # The constructor
    #    @param self The object pointer
    #    @param cameraName The name of the camera in the stereo pair. Ex: /wide_stereo_left
    #    @param outputName The name of the output node
    def __init__(self, cameraName, outputName, transform=None, two_click=False):
        if not transform:
            def t(x):
                return x
            transform = t
        self.transform = transform

        self.name = "%s Viewer" % cameraName
        self.two_click = two_click
        self.cp = False
        self.ch_x = 0
        self.ch_y = 0
        self.x1 = -1
        self.y1 = -1
        self.zoom = 1
        self.offset = (0.0, 0.0)
        self.outputName = outputName
        self.bridge = CvBridge()
        self.create_window()
        self.cameraTopic = cameraName
        self.cameraInfoTopic = "%s/camera_info" % cameraName
        # import pdb; pdb.set_trace()
        self.camera_sub = rospy.Subscriber(
            self.cameraTopic, Image, self.update_background)
        self.camera_info_sub = rospy.Subscriber(
            self.cameraInfoTopic, CameraInfo, self.update_camera_info)

        self.clear_serv = rospy.Service(
            "%s/received" % self.outputName, EmptySrv, self.clear_request)
        self.point_pub = rospy.Publisher(
            self.outputName, String, queue_size=10)
        self.set_listeners()
        self.img = np.zeros((480, 640, 3))

    # Creates a window and updates it for the first time
    def create_window(self):
        cv2.namedWindow(self.name)
        cv2.waitKey(25)
        print "Window created"

    # Sets the background (used for updating the camera stream)
    #    @param background A pointer to the cvImage which will be the background of the window
    def set_background(self, background):
        self.background = background

    # Updates the background, given a new packet of camera data
    #    @param data The camera data (in Image format)
    def update_background(self, data):
        cv_image = self.transform(self.bridge.imgmsg_to_cv2(data))
        # cv2.line(cv_image ,(self.ch_x-25,self.ch_y),(self.ch_x+25,self.ch_y),(255,255,0))
        # cv2.line(cv_image ,(self.ch_x,self.ch_y-25),(self.ch_x,self.ch_y+25),(255,255,0))
        self.img = cv_image

    def update_camera_info(self, data):
        self.set_camera_info(data)

    def set_camera_info(self, info):
        self.camera_info = info

    def set_listeners(self):
        cv2.setMouseCallback(self.name, self.onMouse, 0)
        print "Set Listeners"

    def onMouse(self, event, zoom_x, zoom_y, flags, param):
        self.setCrosshairs(zoom_x, zoom_y)
        (x, y) = self.unZoomPt(zoom_x, zoom_y)

        if event == cv2.EVENT_LBUTTONUP:
            if self.two_click:
                if self.x1 == -1:
                    self.x1 = x
                    self.y1 = y
                else:
                    self.output_point_2(self.x1, self.y1, x, y)
                    self.x1 = -1
                    self.y1 = -1
            else:
                self.output_point(x, y)
                self.setCrosshairs(x, y)
        if event == cv2.EVENT_MBUTTONUP:
            print "Zooming to point (%d,%d)" % (x, y)
            self.zoom *= 2
            self.offset = (x - self.background.width / (2 * self.zoom),
                           y - self.background.height / (2 * self.zoom))
        if event == cv2.EVENT_RBUTTONUP:
            print "Unzooming"
            self.zoom = 1
            self.offset = (0, 0)

    def setCrosshairs(self, x, y):
        self.ch_x = x
        self.ch_y = y

    # Given a pixel on a possibly zoomed window, outputs the proper camera pixel value
    #    @param (zoom_x,zoom_y) The (x,y) coordinates of the click
    def unZoomPt(self, zoom_x, zoom_y):
        scaled_x = zoom_x / self.zoom
        scaled_y = zoom_y / self.zoom
        centered_x = scaled_x + self.offset[0]
        centered_y = scaled_y + self.offset[1]
        return (int(centered_x), int(centered_y))

    # Given a pixel on the camera, outputs the location of that point in terms of the current, possibly zoomed window.
    #    @param (x,y) The (x,y) coordinates of the pixel, in the camera's view
    def zoomPt(self, x, y):
        uncentered_x = x - self.offset[0]
        uncentered_y = y - self.offset[1]
        x = uncentered_x * self.zoom
        y = uncentered_y * self.zoom
        return (int(x), int(y))

    # Publishes the proper point and camera information to the given topic
    #    @param (x,y) The (x,y) coordinates of the pixel, in the camera's view
    def output_point(self, x, y):
        msg = "(%d,%d)" % (x, y)
        self.point_pub.publish(msg)

    def output_point_2(self, x1, y1, x2, y2):
        theta1 = math.pi + math.pi / 2 * np.random.random()
        theta2 = math.pi + math.pi / 2 * np.random.random()
        msg = "(%d,%d,%d,%d,%.4f,%.4f)" % (x1, y1, x2, y2, theta1, theta2)
        print msg
        self.point_pub.publish(msg)

    # The listener, which updates the camera feed and registers onMouse events
    # def listen(self):
    #     try:
    #         # im = cm.jet(self.background / 255)
    #         im = self.background * 16
    #         cv2.imshow(self.name, im)

    #     # if(self.cp != False):
    #     #     cv2.circle(img,self.zoomPt(self.cp.x,self.cp.y),3, (0,255,0),-1)
    #     # cv2.line(img,(self.ch_x-25,self.ch_y),(self.ch_x+25,self.ch_y),(255,255,0))
    #     # cv2.line(img,(self.ch_x,self.ch_y-25),(self.ch_x,self.ch_y+25),(255,255,0))
    #     except AttributeError:
    #         pass

    #     cv2.waitKey(25)

    # Clears the current click point
    def clear_request(self, args):
        self.cp = False
        return []


def usage():
    print "clickwindow.py [name] [cameraName] [outputName]"

# Instantiate a new click_window node


def main(args):
    name = "ClickWindowName"
    rospy.init_node(name)

    def transform(x):
        return 16 * x  # convert 16 bit depth data into image

    def bgr_to_rgb(x):
        return x

    # gui = ClickWindow("/camera/depth/image_rect", "sr300_img_clickpoints", transform)
    gui = ClickWindow("/camera/rgb/image_raw",
                      "sr300_img_clickpoints", bgr_to_rgb)
    print("Pressed Esc to escape")
    while True:
        cv2.imshow(gui.name, gui.img)
        key = cv2.waitKey(25)
        if key == 27:
            print("Exit.")
            break

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        main(args)
    except rospy.ROSInterruptException:
        pass
