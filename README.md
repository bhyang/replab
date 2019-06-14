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

### Configuring Servo Torques
Operating the WidowX with the default torque limits can be hazardous if the arm experiences collisions, especially if the cell is left to collect data autonomously without human supervision. We recommend using a provided script to set all the servo limits to 50% of maximum torque.
```
cd replab_core/scripts
python configure_servos.py
```
Make sure MoveIt! isn't running when using the script. This script does not need to be run again after restarting Docker or powering off the arm.

### Control Noise Compensation
To perform control noise compensation, run
```
rosrun replab_core compute_control_noise.py
```
This will move the arm to 25 predefined points along the floor of the arena and measure the error between the goal position and achieved position. Then, it will visualize these errors and print the parameters for the linear model we use to correct for control noise. These parameters can be entered in `replab_core/src/replab_core/config.py` as `CONTROL_NOISE_COEFFICIENT_ALPHA` and `CONTROL_NOISE_COEFFICIENT_BETA`. This will automatically apply the correction when using `widowx.move_to_grasp()`.

### Robot-Camera Calibration
We prescribe a camera alignment procedure which, once completed, means that you can use the camera-to-robot calibration matrix that is already provided in `replab_core/src/replab_core/config.py`.
The script overlays the view from the camera along with a reference image of our aligned setup to help the user align both views. 
```
cd src/replab_core/scripts
python cam_align.py --ref_image reference.jpg
```
If you have multiple webcams connected, you may need to specify the camera device/path manually using the --cameraA and --cameraB flags (on Ubuntu 16.04, the default path with no other webcams should be /dev/video0 and /dev/video1 for the RGB and depth streams respectively).
```
python cam_align.py --cameraA /dev/video0
```
To align the images, the camera mount needs to be manually adjusted by hand. To further verify that the cell matches ours, we recommend moving the arm to neutral position and aligning both the arm and the camera in the image jointly.

Otherwise, to compute a new calibration matrix, use
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
To label samples, we provide a labeling script that allows the user to manually annotate samples as successes or failures.
```
cd replab_grasping/training
python manual_labeler.py --path [path to data directory]
```

### Training the models

The available models are ``fullimage`` and ``pintogupta``. ``fullimage`` uses both RGB and Depth, while ``pintogupta`` uses only RGB images. Their respective performance is detailed in the REPLAB paper. Pretrained models can be found on the REPLAB website or in the Docker container at `/root/ros_ws/src/replab_grasping/training/models`. The path to the models can be changed in `replab_core/src/replab_core/config.py` by changing `PINTO2016_PRETRAINED_WEIGHTS` and `FULLIMAGE_PRETRAINED_WEIGHTS`.

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

## Reinforcement Learning on REPLAB

This is the code for training and evaluating RL algorithms on REPLAB cells. This is heavily based off of RLkit, available here: https://github.com/vitchyr/rlkit, and a modified Viskit repo, available here: https://github.com/vitchyr/viskit.


### Directory Structure
Files for Reinforcement Learning on REPLAB are located in two places: ``/root/ros_ws/rl_scripts/`` and ``/root/ros_ws/src/replab_rl/``. 

``/root/ros_ws/src/replab_rl/`` contains two folders, ``gym-replab``, a pip package that has the OpenAI Gym Environment for REPLAB and ``src``, which contains a file that allows us to communicate with ROS through Python3. The actual RL scripts are located in ``/root/ros_ws/rl_scripts/``, which contains two folders: ``rlkit`` and ``viskit``. ``rlkit`` contains the base code from the RLkit repository, and modified example scripts for both fixed and randomized reaching tasks. ``viskit`` contains code from the Viskit repository.


### Prerequisites

To train or evaluate a model on a REPLAB cell, you must first run two scripts in the docker container.

In one window, run 

``` sh /root/ros_ws/start.sh ``` to start communicating with the MoveIt commander.

In another, run

```rosrun replab_rl replab_env_subscriber.py``` to enable the Python3 environment to communicate with the Python2 ROS build.

### Training the models

The task is to reach a point in 3d space through controlling the 6 joints of the arm.

There are 4 examples in ``/root/ros_ws/rl_scripts/rlkit/examples``, three is designed for a fixed goal (``{td3, sac, ddpg}.py``) and the other is designed for a randomized goal (``her_td3_gym_fetch_reach.py``).

By default, all of these use the GPU. If you aren't running this with a GPU, please change ``ptu.set_gpu_mode(True)`` to ``ptu.set_gpu_mode(False)`` near the bottom of the example files.

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

Note: by default, the data directory containing parameters and stats are saved in ``/root/ros_ws/rl_scripts/rlkit/data/[NAME]/[DATE_TIME]``

### Evaluating a Policy

The policy is evaluated at every epoch during training (and this data is saved), however you can also manually evaluate a saved policy.

Example scripts for evaluating a policy are in ``/root/ros_ws/rl_scripts/rlkit/scripts``. For example, to visualize a policy on the real robot, run:

```
cd /root/ros_ws/rl_scripts/rlkit

source activate rlkit

python scripts/[POLICY_SCRIPT].py --[args specified in script] [path_to_params.pkl]
```