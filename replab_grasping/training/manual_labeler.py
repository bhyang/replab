import h5py

import numpy as np
import matplotlib.pyplot as plt

import sys, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
parser.add_argument('--skip_opens', type=int, default=0)
parser.add_argument('--start', type=int, default=0)

args = parser.parse_args()

plt.ion()

fig, ax = plt.subplots(2, 2)

im_rgb_before = ax[0, 0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
im_depth_before = ax[0, 1].imshow(np.zeros((480, 640), dtype=np.uint16))
im_rgb_after = ax[1, 0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
im_depth_after = ax[1, 1].imshow(np.zeros((480, 640), dtype=np.uint16))

im_depth_before.set_clim(0, 700)
im_depth_after.set_clim(0, 700)


def plot_images(rgb_before, depth_before, rgb_after, depth_after):
	im_rgb_before.set_data(rgb_before)
	im_depth_before.set_data(depth_before)
	im_rgb_after.set_data(rgb_after)
	im_depth_after.set_data(depth_after)

src = args.path + '/'

file_ids = [int(file[:-5]) for file in os.listdir(src) if file[-5:] == '.hdf5']
file_ids.sort()
file_ids = [str(i) + '.hdf5' for i in file_ids if i >= args.start]

for file in file_ids:
	print('Labeling %s' % file)
	with h5py.File(src + file, 'r+') as fl:
		before = fl['before_img'].value
		after = fl['after_img'].value
		closure = fl['gripper_closure'].value
		existing_label = 'success' in fl.keys()

		if args.skip_opens and closure > .005:
			user_in = 'hit'
		else:
			plot_images(before[:,:,:3].astype(np.uint8), before[:,:,3], after[:,:,:3].astype(np.uint8), after[:,:,3])
			print('Closure: %.3f' % closure)
			user_in = raw_input('Empty for miss, q for quit, otherwise any for hit: ')
		if user_in == '':
			print('Miss')
			if existing_label:
				fl['success'][...] = False
			else:
				fl['success'] = False
		elif user_in != 'q':
			print('Hit')
			if existing_label:
				fl['success'][...] = True
			else:
				fl['success'] = True
		else:
			print('Quitting')
			exit()
