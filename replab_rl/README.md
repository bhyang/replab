# RL on REPLAB (Work in Progress)

This is the code for training and evaluating RL algorithms on REPLAB cells. This is heavily based off of RLKit, available here: https://github.com/vitchyr/rlkit, and a modified Viskit repo, available here: https://github.com/vitchyr/viskit.


### Directory Structure

This directory only contains a pip package that has the OpenAI Gym Environment for REPLAB and a file in the ``src`` directory that allows us to communicate with ROS through Python3. The actual RL scripts are located in ``/root/ros_ws/rl_scripts/``, which contains two folders: ``rlkit`` and ``viskit``. ``rlkit`` contains the base code from the RLKit repository, and modified example scripts for both fixed and randomized reaching tasks. ``viskit`` contains code from the Viskit repository.


### Prerequisites

To train or evaluate a model on a REPLAB cell, you must first run two scripts in the docker container.

In one window (or tmux session), run 

``` sh /root/ros_ws/start.sh ``` to start communicating with the MoveIt commander.

In another, run

```rosrun replab_rl replab_env_subscriber.py``` to enable the Python3 environment to communicate with the Python2 ROS build.

### Training the models

There are 2 examples in ``/root/ros_ws/rl_scripts/rlkit/examples``, one is designed for a fixed goal (``td3.py``) and the other isn't (``her/her_td3_gym_fetch_reach.py``).

To get started, run


```
cd /root/ros_ws/rl_scripts/rlkit/

source activate rlkit

python examples/[EXAMPLE_FILE].py
```

For each of these example scripts, the parameters and hyperparameters are easily adjustable by modifying them directly in the file.

For the fixed goal environments, you can modify the fixed goal by directly modifying ``/root/ros_ws/src/replab_rl/gym_replab/gym_replab/envs/replab_env.py``


### Visualizing results

We use Viskit to visualize the results. Run

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

python scripts/[POLICY_SCRIPT].py
```
