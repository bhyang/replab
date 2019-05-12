from __future__ import division

import numpy as np
import h5py
import os
import sys
import time
import cv2

def main():
    fileList = []
    counter = 0
    paths = [] # list data directories here
    for path in paths:
        for f in os.listdir(path):
            if f[-5:] == '.hdf5':
                fileList.append(path + f)

    grasps = []
    successes = []
    angles = []
    crops = []
    widths = []

    print('Processing %d samples' % len(fileList))
    
    for i in range(len(fileList)):
        print(i, fileList[i])
        with h5py.File(fileList[i], 'r') as f:
            try:
                before = f['before_img'].value
                # after = f['after_img'].value

                pose = f['pose'].value
                
                joints = f['joints'].value
                grasp = np.concatenate([pose[:3], [joints[4]]], axis=0)

                success = f['success'].value
                width = f['gripper_closure'].value

                angle = joints[0] + joints[4]
                widths.append(width)
                grasps.append(grasp)
                successes.append(success)
                crops.append(f['pixel_point'].value)
                angles.append(angle)
                np.save('before/' + str(i), before)
                # np.save('after/' + str(counter), after)
                counter += 1
            except:
                print('Skipping')
                continue


    np.save('grasps', np.array(grasps))
    np.save('successes', np.array(successes))
    np.save('angles', np.array(angles))
    np.save('filelist', np.array(fileList))
    np.save('widths', np.array(widths))
    np.save('crops', np.array(crops))
    
if __name__ == "__main__":
    main()
