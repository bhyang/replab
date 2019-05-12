#!/usr/bin/env python

import numpy as np
import rospy
import time
import argparse
from executor import *
from policy import *
from replab_core.config import *
from replab_core.utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=60,
                        help="Number of evaluation trials")
    parser.add_argument('--datapath', type=str,
                        default='', help="Path for saving data samples")
    parser.add_argument('--save', type=int, default=1,
                        help="Toggles whether samples are saved")
    parser.add_argument('--start', type=int, default=0,
                        help="Starting sample index for sample numbering (useful for resuming interrupted evaluations)")
    parser.add_argument('--method', type=str, default='datacollection',
                        help="Method used for planning grasps", choices=METHODS)

    args = parser.parse_args()

    assert args.method in METHODS

    rospy.init_node("executor_widowx")
    executor = Executor(scan=False, datapath=args.datapath, save=args.save)

    executor.widowx.move_to_neutral()
    executor.widowx.open_gripper()

    objects_picked = 0
    running_misses = 0

    rospy.sleep(1)

    start = time.time()

    if args.method == 'principal-axis':
        policy = PrincipalAxis()
    elif args.method == 'pinto2016':
        policy = Pinto2016(PINTO2016_PRETRAINED_WEIGHTS)
    elif args.method == 'datacollection':
        policy = DataCollection(noise=True)
    elif args.method == 'datacollection-noiseless':
        policy = DataCollection(noise=False)
    elif args.method == 'fullimage':
        policy = FullImage(FULLIMAGE_PRETRAINED_WEIGHTS)
    else:
        print('Method not recognized, exiting')
        exit()

    sample_id = args.start

    while sample_id < args.start + args.trials:

        print('\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n')
        print('Grasp %d' % sample_id)

        rgbd, pc = executor.get_rgbd(), executor.get_pc()

        executor.sample['filtered_pc'] = pc

        try:
            grasps = policy.plan_grasp(rgbd, pc)
        except ValueError as ve:
            traceback.print_exc(ve)
            print('Error planning, resetting...')
            executor.widowx.move_to_neutral()
            executor.widowx.open_gripper()
            continue

        if len(grasps) == 0:
            print('No grasps plannable, sweeping rig')
            executor.widowx.sweep_arena()
            executor.widowx.move_to_neutral()
            executor.widowx.open_gripper()
            continue

        confidences = []
        kept_indices = []

        for i, (grasp, confidence) in enumerate(grasps):
            if inside_polygon(grasp[:3], END_EFFECTOR_BOUNDS):
                kept_indices.append(i)
                confidences.append(confidence)

        if len(confidences) == 0:
            print('All planned grasps out of bounds / invalid, resetting...')
            executor.widowx.move_to_neutral()
            executor.widowx.open_gripper()
            continue

        selected = np.random.choice(np.argsort(confidences)[-5:])
        grasp = grasps[kept_indices[selected]][0]

        success, err = executor.execute_grasp(grasp, manual_label=True)

        if err:
            executor.widowx.move_to_neutral()
            executor.widowx.open_gripper(drop=True)
            continue

        if args.save:
            executor.save_sample(sample_id)

        print('Success: %r' % success)

        if success == 1:
            running_misses = 0
            objects_picked += 1
            executor.widowx.discard_object()
        elif success == 0:
            running_misses += 1
            executor.widowx.open_gripper(drop=True)
        else:
            print('Discarding sample and redoing grasp')
            executor.widowx.move_to_neutral()
            executor.widowx.open_gripper()
            continue

        executor.evaluation_data.append(objects_picked)

        executor.widowx.move_to_neutral()

        executor.widowx.open_gripper()

        if running_misses > 10:
            print('10 misses exceeded -- sweeping arena')
            running_misses = 0
            executor.widowx.sweep_arena()
            executor.widowx.move_to_neutral()

        if objects_picked == 20:
            print('All objects picked. Terminating...')

        sample_id += 1

        rospy.sleep(1.5)

    end = time.time()
    print('Time elapsed : %.2f' % (end - start))
    print('Done')

if __name__ == '__main__':
    main()
