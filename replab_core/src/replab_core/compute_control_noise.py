#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from controller import WidowX


def main():
    rospy.init_node("control_noise")
    widowx = WidowX()

    widowx.move_to_neutral()

    targets = []
    actual = []

    for x in np.linspace(-.14, .14, 5):
        for y in np.linspace(-.14, .14, 5):
            if x == 0. and y == 0.:
                continue

            widowx.orient_to_target(x=x,y=y)
            widowx.move_to_grasp(x,y,.42,0)

            rospy.sleep(.25)

            pose = widowx.get_current_pose().pose
            targets.append((x,y))
            actual.append((pose.position.x, pose.position.y))

            widowx.move_to_drop()

    targets = np.array(targets)
    actual = np.array(actual)

    plt.title('Control Noise Error')
    plt.scatter(targets[:,0], targets[:,1], color='red', label='target')
    plt.scatter(actual[:,0], actual[:,1], color='blue', label='actual')
    plt.legend()
    plt.show()

    # compute noise coefficient
    targets = targets.reshape(-1,1)
    actual = actual.reshape(-1,1)
    targets = np.concatenate([targets, np.ones_like(targets)], axis=1)
    x, _, _, _ = np.linalg.lstsq(targets, actual, rcond=None)
    print('Alpha: %.4f' % x[0].item())
    print('Beta: %.4f' % x[1].item())

if __name__ == '__main__':
    main()
