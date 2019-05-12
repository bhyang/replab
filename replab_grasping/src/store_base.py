#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospy
import time
import argparse
from executor import *
from replab_core.config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100,
                        help="Number of iterations to collect")
    parser.add_argument('--plot', action="store_true", default=True)
    args = parser.parse_args()

    rospy.init_node("executor_widowx")
    executor = Executor(scan=True)
    executor.widowx.move_to_neutral()

    rospy.sleep(1)

    print('Scanning base...')
    executor.scan_base(args.iterations)

    print('Plotting base pointcloud...')
    pc = executor.base_pc
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[1:, 0], pc[1:, 1], pc[1:, 2])
    plt.show()

    print('Done')

if __name__ == '__main__':
    main()
