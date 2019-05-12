import gym
from gym import error, spaces, utils
from gym.utils import seeding
from numbers import Number

from collections import OrderedDict

import numpy as np


import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String

import random


class ReplabEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        How to initialize this environment:
        env = gym.make('replab-v0')._start_rospy(goal_oriented=[GOAL_ORIENTED])
        If goal_oriented is true, then the environment's observations become a dict
        and the goal is randomly resampled upon every reset

        params:
        goal_oriented: Changes some aspects of the environment for goal-oriented tasks

        rospy.init_node is set with random number so we can have multiple
        nodes running at the same time.

        self.goal is set to a hardcoded goal if goal_oriented = False
        """
        self.obs_space_low = np.array(
            [-.16, -.15, 0.14, -3.1, -1.6, -1.6, -1.8, -3.1, 0])
        self.obs_space_high = np.array(
            [.16, .15, .41, 3.1, 1.6, 1.6, 1.8, 3.1, 0.05])
        observation_space = spaces.Box(
            low=self.obs_space_low, high=self.obs_space_high, dtype=np.float32)
        self.observation_space = observation_space
        self.action_space = spaces.Box(low=np.array([-0.5, -0.25, -0.25, -0.25, -0.5, -0.005]) / 5,
                                       high=np.array([0.5, 0.25, 0.25, 0.25, 0.5, 0.005]) / 5, dtype=np.float32)
        self.current_pos = None
        #self.goal = np.array([-.14, -.13, 0.26])
        self.goal = self.sample_goal_for_rollout()
         
    def _start_rospy(self, goal_oriented=False):
        self.rand_init = random.random()
        rospy.init_node("widowx_custom_controller_%0.5f" % self.rand_init)
        self.reset_publisher = rospy.Publisher(
            "/replab/reset", String, queue_size=1)
        self.position_updated_publisher = rospy.Publisher(
            "/replab/received/position", String, queue_size=1)
        self.action_publisher = rospy.Publisher(
            "/replab/action", numpy_msg(Floats), queue_size=1)
        self.current_position_subscriber = rospy.Subscriber(
            "/replab/action/observation", numpy_msg(Floats), self.update_position)
        rospy.sleep(2)
        self.goal_oriented = goal_oriented
        if self.goal_oriented:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=np.array(
                    [-.16, -.15, 0.25]), high=np.array([.16, .15, 0.41]), dtype=np.float32),
                achieved_goal=spaces.Box(low=self.obs_space_low[
                    :3], high=self.obs_space_high[:3], dtype=np.float32),
                observation=self.observation_space
            ))
        self.reset()
        return self

    def update_position(self, x):
        self.current_pos = np.array(x.data)
        self.position_updated_publisher.publish('received')

    def sample_goal_for_rollout(self):
        return np.random.uniform(low=np.array([-.14, -.13, 0.26]), high=np.array([.14, .13, .39]))

    def set_goal(self, goal):
        self.goal = goal
        print(self.goal)

    def step(self, action):
        """

        Parameters
        ----------
        action : [change in x, change in y, change in z]

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                either current position or an observation object, depending on
                the type of environment this is representing
            reward (float) :
                negative, squared, l2 distance between current position and 
                goal position
            episode_over (bool) :
                Whether or not we have reached the goal
            info (dict) :
                 For now, all this does is keep track of the total distance from goal.
                 This is used for rlkit to get the final total distance after evaluation.
                 See function get_diagnostics for more info.
        """
        action = np.array(action, dtype=np.float32)
        #target_pos = np.add(action, self.current_pos)
        #self.task_finished = False
        self.action_publisher.publish(action)
        self.current_pos = np.array(rospy.wait_for_message(
            "/replab/action/observation", numpy_msg(Floats)).data)

        reward = self._get_reward(self.goal)

        episode_over = False
        total_distance_from_goal = np.sqrt(-reward)

        info = {}
        info['total_distance'] = total_distance_from_goal

        if reward > -0.0001:
            episode_over = True

        if self.goal_oriented:
            obs = self._get_obs()
            return obs, reward, episode_over, info

        return self.current_pos, reward, episode_over, info

    def reset(self):
        self.reset_publisher.publish(String("RESET"))
        self.current_pos = np.array(rospy.wait_for_message(
            "/replab/action/observation", numpy_msg(Floats)).data)
        if self.goal_oriented:
            self.set_goal(self.sample_goal_for_rollout())
            return self._get_obs()
        return self.current_pos

    def _get_obs(self):
        obs = {}
        obs['observation'] = self.current_pos
        obs['desired_goal'] = self.goal
        obs['achieved_goal'] = self.current_pos[:3]
        return obs

    def sample_goals(self, num_goals):
        sampled_goals = np.array(
            [self.sample_goal_for_rollout() for i in range(num_goals)])
        goals = {}
        goals['desired_goal'] = sampled_goals
        return goals

    def _get_reward(self, goal):
        return - (np.linalg.norm(self.current_pos[:3] - goal) ** 2)

    def render(self, mode='human', close=False):
        pass

    def compute_reward(self, achieved_goal, goal, info):
        return - (np.linalg.norm(achieved_goal - goal)**2)

    def get_diagnostics(self, paths):
        """
        This adds the diagnostic "Final Total Distance"
        """
        def get_stat_in_paths(paths, dict_name, scalar_name):
            if len(paths) == 0:
                return np.array([[]])

            if type(paths[0][dict_name]) == dict:
                # Support rllab interface
                return [path[dict_name][scalar_name] for path in paths]
            return [[info[scalar_name] for info in path[dict_name]] for path in paths]

        def create_stats_ordered_dict(
                name,
                data,
                stat_prefix=None,
                always_show_all_stats=True,
                exclude_max_min=False,
        ):
            if stat_prefix is not None:
                name = "{} {}".format(stat_prefix, name)
            if isinstance(data, Number):
                return OrderedDict({name: data})

            if len(data) == 0:
                return OrderedDict()

            if isinstance(data, tuple):
                ordered_dict = OrderedDict()
                for number, d in enumerate(data):
                    sub_dict = create_stats_ordered_dict(
                        "{0}_{1}".format(name, number),
                        d,
                    )
                    ordered_dict.update(sub_dict)
                return ordered_dict

            if isinstance(data, list):
                try:
                    iter(data[0])
                except TypeError:
                    pass
                else:
                    data = np.concatenate(data)

            if (isinstance(data, np.ndarray) and data.size == 1
                    and not always_show_all_stats):
                return OrderedDict({name: float(data)})

            stats = OrderedDict([
                (name + ' Mean', np.mean(data)),
                (name + ' Std', np.std(data)),
            ])
            if not exclude_max_min:
                stats[name + ' Max'] = np.max(data)
                stats[name + ' Min'] = np.min(data)
            return stats
        statistics = OrderedDict()
        stat_name = 'total_distance'
        stat = get_stat_in_paths(paths, 'env_infos', stat_name)
        statistics.update(create_stats_ordered_dict('Final %s' % (stat_name), [
                          s[-1] for s in stat], always_show_all_stats=True,))
        return statistics

    # Functions for pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['reset_publisher']
        del state['position_updated_publisher']
        del state['action_publisher']
        del state['current_position_subscriber']
        return state

    def __setstate__(self, state):
        self.reset_publisher = rospy.Publisher(
            "/replab/reset", String, queue_size=1)
        self.position_updated_publisher = rospy.Publisher(
            "/replab/received/position", String, queue_size=1)
        self.action_publisher = rospy.Publisher(
            "/replab/action", numpy_msg(Floats), queue_size=1)
        self.current_position_subscriber = rospy.Subscriber(
            "/replab/action/observation", numpy_msg(Floats), self.update_position)
        self.__dict__.update(state)
        # self._start_rospy(goal_oriented=state['goal_oriented'])
        self.reset()
