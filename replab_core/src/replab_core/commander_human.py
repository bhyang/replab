#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from sensor_msgs.msg import (
    Image,
    PointCloud2
)
from std_msgs.msg import (
    UInt16,
    Int64,
    Float32,
    Header
)

from click_window import ClickWindow

import cv2
import numpy as np
from itertools import product

action_publisher = None


class CommanderHuman():

    def __init__(self, init_X=-1, init_Y=-1, tol=1):
        self.X = init_X
        self.Y = init_Y

    def on_click(self, data):
        self.X, self.Y = eval(data.data)

        msg = "(%d,%d)" % (self.X, self.Y)

        action_publisher.publish(msg)
        print(msg)

    def get_on_request(self, action_publisher):
        def on_request(self, data):
            theta = math.pi + np.random.random() * math.pi / 2
            msg = "(%d,%d,%.4f)" % (self.X, self.Y, theta)
            action_publisher.pub(msg)


def main():
    name = "CommanderHuman"
    rospy.init_node(name)

    gui = ClickWindow("/camera/rgb/image_raw", "clicked_point")
    commander_human = CommanderHuman(240, 320, 1)

    # Register subscribers
    global action_publisher
    click_subscriber = rospy.Subscriber(
        '/clicked_point', String, commander_human.on_click)
    action_publisher = rospy.Publisher('/target_action', String, queue_size=1)
    request_subscriber = rospy.Subscriber(
        '/action_request', Float32, commander_human.get_on_request(action_publisher))

    while True:
        cv2.imshow(gui.name, gui.img)
        key = cv2.waitKey(25)
        if key == 27:
            print("Exit.")
            break


if __name__ == '__main__':
    main()
