#!/usr/bin/env python

import numpy as np
import rospy
import argparse
from click2control import Click2Control
from sklearn.linear_model import LinearRegression


def compute_calibration(robot_points, camera_points):
    lr = LinearRegression().fit(camera_points, robot_points)
    predicted = lr.predict(camera_points)
    residuals = np.abs(predicted - robot_points)

    co = lr.coef_
    trans = lr.intercept_
    tf_matrix = np.matrix([[co[0, 0],	co[0, 1],	co[0, 2],	trans[0]],
                           [co[1, 0],	co[1, 1],	co[1, 2],	trans[1]],
                           [co[2, 0],	co[2, 1],	co[2, 2],	trans[2]],
                           [0.0,		0.0,		0.0,		1.0		]])
    return tf_matrix, residuals


def main():
    parser = argparse.ArgumentParser(
        description="Allows user to perform arm-camera calibration locally using a GUI window")
    args = parser.parse_args()

    rospy.init_node("calibration")

    executor = Click2Control(execute=False, calibrate=True)

    rospy.sleep(2)

    print('Ctrl-C to finish gathering correspondences')
    rospy.spin()

    calibration_matrix, residuals = compute_calibration(
        executor.robot_coordinates, executor.camera_coordinates)

    print('Residuals (cm):')
    print(residuals * 100.)

    print('Calibration matrix:')
    print(calibration_matrix)

if __name__ == '__main__':
    main()
