# REPLAB: A Reproducible Low-Cost Arm Benchmark for Robotic Learning
![Imgur](https://i.imgur.com/rf2tucH.jpg)
This package contains the scripts used to operate a REPLAB cell. More details and the full paper can be found [here](https://sites.google.com/view/replab/).

## Setup
Since the package is meant to be run inside of the provided Docker container, no additional installation setup is required aside from running the Docker image (instructions for setting up a Docker container can be found on the website).

*Note: certain graphical components for scripts (e.g. calibration, click2window) may not be displayed without using nvidia-docker. However, most scripts (including data collection and evaluation) do not require the use of a graphical interface.*

### Initializing the Camera / MoveIt!
Whenever operating the cell, the camera nodelet and MoveIt! stack need to be initialized before use. A convenience script `start.sh` is provided in `/root/ros_ws/` within the Docker container for this purpose.
```
sh /root/ros_ws/start.sh
```
To run manually, use
```
roslaunch realsense_camera sr300_nodelet_rgbd.launch
roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=false
```
These scripts will run in the background during the operation of the cell. Once this script is running, we recommend entering the container in another terminal using `docker exec`.
```
docker exec -it [container ID] bash
```
The container ID can be found using `docker container ls`.

### Controller Interface
To operate the arm by directly issuing commands, use
```
rosrun replab_core controller.py
```
This will launch a `pdb` command line interface where the user can freely use predefined motion routines to control the arm. Example usage includes:
```
widowx.move_to_neutral()  # Moves the arm to neutral position

widowx.get_joint_values() # Returns the servo positions in joint angles

widowx.move_to_drop()     # Moves the arm to object-dropping position
widowx.move_to_drop(-.5)  # Moves the arm to object-dropping position with the first servo rotated -.5 radians from neutral

widowx.open_gripper()     # Opens the gripper
widowx.close_gripper()    # Closes the gripper

widowx.sweep_arena()      # Sweep the arena

NEUTRAL_VALUES[0] += .5   # Modify the position of the first servo in the neutral position by rotating it .5 radians
widowx.move_to_neutral()  # Move to the new neutral position
```

### Robot-Camera Calibration
If the camera is aligned to our provided reference image and the cell is built to specification, then calibration is not required since our provided calibration matrix (which is already stored in `replab_core/src/replab_core/config.py`) should work out of the box. Otherwise, to compute a new calibration matrix, use
```
rosrun replab_core commander_human.py
```
This will launch a GUI showing the input from the camera. In a separate window, run
```
rosrun replab_core calibrate.py
```
The script works by collecting correspondences between the position of the end-effector in robot coordinates and the position in camera coordinates, which are used to compute a calibration matrix. Simply click a point in the GUI to save the camera coordinate of the clicked point. Then, move the arm to the clicked point and record the position of the end-effector. Note that the script requires the end-effector to be oriented downward towards the arena floor for each correspondence. We recommend collecting at least 15 correspondences around the arena. Once finished (ctrl-C to exit), the script will output the computed calibration matrix that can be copied into `replab_core/src/replab_core/config.py`.

### Click2Control
To verify the calibration and the construction of the cell, you can use `click2control.py` which execute user-specified grasps from the `commander_human.py` GUI
```
rosrun replab_core click2control.py
```
If the executed grasps don't line up with the user-specified points, then you may need to recheck the camera-arm calibration, either by adjusting the position of the arm/camera or recomputing the calibration matrix.

## Grasping
### Collecting the Point Cloud Base
The grasping routine is reliant on blob detection to identify objects in the point cloud. This requires a base of point cloud points to perform background subtraction and filter out non-object points. To collect the base point cloud, clear the arena of any objects and use
```
rosrun replab_grasping store_base.py
```
The script will move the arm to neutral position at the start. This script will take a few minutes to run. The point cloud is stored in `pc_base.npy`. 

*Note: the base point cloud is sensitive to camera/arena peturbations, so this process may need to be repeated every so often to recollect the point cloud base.*

### Data Collection
Before starting data collection, make sure there are objects in the arena. Then use
```
rosrun replab_grasping collect_data.py --samples [# of samples] --datapath [path for saving samples]
```
If there are issues with blob detection, considering tuning the `DBSCAN_EPS` and `DBSCAN_MIN_SAMPLES` parameters in `replab_core/src/replab_core/config.py`.

### Training the models

The available models are ``fullimage`` and ``pintogupta``. ``fullimage`` uses both RGB and Depth, while ``pintogupta`` uses only RGB images. Their respective performance is detailed in the REPLAB paper. 

To train a model, run

```
python train.py --batch_size [batch size] --epochs [number of epochs to train for] --lr [learning rate] --method [fullimage or pintogupta] --resultpath [where the results are saved] --datapath [directory where the data is stored (including the .npy files and the 'before' directory] --start [epoch to start training from (used for resuming training)] --weight_decay [weight decay for Adam optimizer]
```

### Visualizing results

We use tensorboard to visualize the results. Simply run:

```
tensorboard --logdir [RESULT_PATH]
```

### Evaluation
Before running an evaluation, make sure there are test objects in the arena. Then use
```
rosrun replab evaluation.py --method [method name] --datapath [path for saving samples]
```

## RL

This is the code for training and evaluating RL algorithms on REPLAB cells. This is heavily based off of RLKit, available here: https://github.com/vitchyr/rlkit, and a modified Viskit repo, available here: https://github.com/vitchyr/viskit.


### Directory Structure
Reinforcement Learning on REPLAB files are located in two places: ``/root/ros_ws/rl_scripts/`` and ``/root/ros_ws/src/replab_rl/``. 

``/root/ros_ws/src/replab_rl/`` contains two folders, ``gym-replab``, a pip package that has the OpenAI Gym Environment for REPLAB and ``src``, which contains a file that allows us to communicate with ROS through Python3. The actual RL scripts are located in ``/root/ros_ws/rl_scripts/``, which contains two folders: ``rlkit`` and ``viskit``. ``rlkit`` contains the base code from the RLKit repository, and modified example scripts for both fixed and randomized reaching tasks. ``viskit`` contains code from the Viskit repository.


### Prerequisites

To train or evaluate a model on a REPLAB cell, you must first run two scripts in the docker container.

In one window, run 

``` sh /root/ros_ws/start.sh ``` to start communicating with the MoveIt commander.

In another, run

```rosrun replab_rl replab_env_subscriber.py``` to enable the Python3 environment to communicate with the Python2 ROS build.

### Training the models

There are 2 examples in ``/root/ros_ws/rl_scripts/rlkit/examples``, one is designed for a fixed goal (``td3.py``) and the other is designed for a randomized goal (``her/her_td3_gym_fetch_reach.py``).

To get started, run


```
cd /root/ros_ws/rl_scripts/rlkit/

source activate rlkit

python examples/[EXAMPLE_FILE].py
```

For each of these example scripts, the parameters and hyperparameters are easily adjustable by modifying them directly in the file.

For the fixed goal environments, you can modify the fixed goal by directly modifying ``/root/ros_ws/src/replab_rl/gym_replab/gym_replab/envs/replab_env.py``


### Visualizing results

RLkit recommends Viskit to visualize the results. To view them, run:

```
source activate rlkit

python /root/ros_ws/rl_scripts/viskit/viskit/frontend.py [DATA_DIRECTORY]
```

Then, in your browser, navigate to the IP address of the docker container and the port listed by viskit.

Note: by default, the data directory containing parameters and stats are saved in ``/root/ros_ws/rl_scripts/rlikit/data/[NAME]/[DATE_TIME]``

### Evaluating a Policy

Example scripts for evaluating a policy are in ``/root/ros_ws/rl_scripts/rlkit/scripts``. Run:

```
cd /root/ros_ws/rl_scripts/rlkit

source activate rlkit

python scripts/[POLICY_SCRIPT].py --[args specified in script]
```