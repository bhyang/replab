#!/usr/bin/env python

import numpy as np
import cv2
import argparse
from scipy.misc import imread

ACTIVE_CAMERAS = []


def main():
    parser = argparse.ArgumentParser(
        description="Calibration script for manually aligning cameras")
    parser.add_argument('--ref_image', type=str,
                        help='Use static reference image to calibrate instead of streaming from two sources. Image should be 640x480 RGB')
    parser.add_argument('--cameraA', type=str,
                        help='Specify device/path to use for camera A', default='/dev/video0')
    parser.add_argument('--cameraB', type=str,
                        help='Specify device/path to use for camera B', default='/dev/video2')

    args = parser.parse_args()

    print('Initializing video capture...')

    cameraA = cv2.VideoCapture(args.cameraA)
    assert cameraA.read()[
        0], 'Camera A returned None, check that the device/path is correct and that the camera is connected'
    ACTIVE_CAMERAS.append(cameraA)

    try:
        if args.ref_image:
            try:
                img = imread(args.ref_image)
            except IOError as ie:
                print(
                    'Reference image (%s) invalid / not found, check that the path is correct' % args.ref_image)
                exit()
            assert img.shape == (
                480, 640, 3), 'Invalid image dimensions'
            assert img.dtype == np.uint8, 'Invalid image dtype, requires uint8'

            print('Streaming images from A only, press (q) to quit')

            while(True):
                ret, frame = cameraA.read()

                frame = (frame.astype(np.uint16) + img.astype(np.uint16)) / 2
                frame = frame.astype(np.uint8)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            cameraB = cv2.VideoCapture(args.cameraB)
            assert cameraB.read()[
                0], 'Camera B returned None, check that the device/path is correct and that the camera is connected.'
            ACTIVE_CAMERAS.append(cameraB)

            print('Streaming images from A and B, press (q) to quit')

            while(True):
                # Capture frame-by-frame
                ret, frameA = cameraA.read()
                ret, frameB = cameraB.read()

                frame = (frameA.astype(np.uint16) +
                         frameB.astype(np.uint16)) / 2
                frame = frame.astype(np.uint8)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        print('Cleaning up...')
        for camera in ACTIVE_CAMERAS:
            camera.release()
        cv2.destroyAllWindows()
        print('Done')

if __name__ == '__main__':
    main()
