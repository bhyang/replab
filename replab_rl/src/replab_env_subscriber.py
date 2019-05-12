#!/usr/bin/env python
import rospy
from replab_core.controller import *
from replab_core.config import *
import numpy as np

import rospy
from std_msgs.msg import String
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

widowx = None


def activate_widowx():
    global widowx
    widowx = WidowX(True)


observation_publisher = rospy.Publisher(
    "/replab/action/observation", numpy_msg(Floats), queue_size=1)
joint_names = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'gripper_joint']
rospy.sleep(2)


def get_state():
    pos = widowx.get_current_pose().pose.position
    joints = widowx.joint_state
    joint_dict = dict(zip(joints.name, joints.position))
    joints = [joint_dict[name] for name in joint_names]
    pos = [pos.x, pos.y, pos.z]
    pos.extend(joints)
    return pos


def get_reward(goal):
    """ Reward is negative L2 distance from objective, squared """
    return -(np.linalg.norm(np.array(goal) - np.array(get_state()))**2)


def take_action(data):
    """
    Publishes [current x, current y, current z]
    """
    action = data.data
    #current_pos = np.array(get_state(), dtype=np.float32)
    #goal = np.add(action, current_pos)
    #widowx.move_to_position(float(goal[0]), float(goal[1]), float(goal[2]))
    widowx.move_to_joint_position(action)
    # widowx.move_to_position_and_joint(action)
    current_state = np.array(get_state(), dtype=np.float32)
    task_finished = False
    observation_publisher.publish(current_state)


def reset(data):
    widowx.move_to_reset()
    rospy.sleep(0.5)
    observation_publisher.publish(np.array(get_state(), dtype=np.float32))


def listener():
    action_subscriber = rospy.Subscriber(
        "/replab/action", numpy_msg(Floats), take_action)
    reset_subscriber = rospy.Subscriber("/replab/reset", String, reset)
    activate_widowx()
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node('replab_gym_node')
    listener()
