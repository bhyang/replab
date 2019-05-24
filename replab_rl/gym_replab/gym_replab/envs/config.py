import numpy as np
NEUTRAL_VALUES = [0.015339807878856412, -1.4839419194602816,
                  1.4971652489763858, -0.008369006790373335, -0.08692557798018634]

# RL BOUNDS
BOUNDS_FLOOR = .41
BOUNDS_LEFTWALL = .14
BOUNDS_RIGHTWALL = -.14
BOUNDS_FRONTWALL = -.13
BOUNDS_BACKWALL = .13

JOINT_MIN = np.array([
    -3.1,
    -1.571,
    -1.571,
    -1.745,
    -2.617,
    0.003
])
JOINT_MAX = np.array([
    3.1,
    1.571,
    1.571,
    1.745,
    2.617,
    0.03
])
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'gripper_joint']
SIM_START_POSITION = np.array([-0.185033226409, 0.00128528, 0.46227163])