import numpy as np
import gym
import copy
from gym import spaces


class ReinforcementTestEnv(gym.Env):
    def __init__(self, action_size=4, desired=1, sequence_desired=False, discrete=False,
                 reward_mode=None, reward_scalar=1, max_undesired_steps=None):
        self.action_size = action_size
        self.desired = desired
        # self.desired = np.full(act_shape, desired)
        self.sequence_desired = sequence_desired
        self.discrete = discrete
        self.reward_mode = reward_mode
        self.reward_scalar = reward_scalar
        self.max_undesired_steps = max_undesired_steps

        if discrete:
            self.action_space = spaces.Discrete(self.action_size)
        else:
            act_shape = (1,)
            # act_shape = (action_size,)
            self.action_space = spaces.Box(np.full(act_shape, -np.inf, np.float32),
                                           np.full(act_shape, np.inf, np.float32))
        self.obs_shape = (1,)
        self.obs_dtype = np.int32 if discrete else np.float32
        self.observation_space = spaces.Box(np.full(self.obs_shape, -np.inf, self.obs_dtype),
                                            np.full(self.obs_shape, np.inf, self.obs_dtype))
        self.state = np.zeros(self.obs_shape, self.obs_dtype)
        self.desired_index = 0

    def is_reward_mode(self, mode):
        return self.reward_mode is not None and mode in self.reward_mode

    def reset(self):
        self.undesired_steps = 0
        # self.state = np.array(np.random.rand(*self.obs_shape) * 100).astype(self.obs_dtype)
        self.state = np.zeros(self.obs_shape, self.obs_dtype)
        self.desired_index = 0
        return copy.deepcopy(self.state)

    def step(self, action):
        reward = 0
        done = False

        if self.discrete:
            self.state[0] = action

            # check if desired action has been performed
            if isinstance(self.desired, list):
                if self.sequence_desired:
                    # must sent desired in sequence
                    is_next_desired = action == self.desired[self.desired_index]
                    if is_next_desired:
                        self.desired_index = (self.desired_index + 1) % len(self.desired)
                        if self.is_reward_mode("sparse"):
                            is_desired = self.desired_index == 0
                        else:
                            is_desired = is_next_desired
                    else:
                        is_desired = False
                        self.desired_index = 0
                else:
                    is_desired = action in self.desired
            else:
                is_desired = action == self.desired

            if self.is_reward_mode("decaying"):
                reward = -1
                if is_desired:
                    reward = self.action_size * self.reward_scalar
                    done = True
            elif self.is_reward_mode("sparse"):
                if is_desired:
                    reward = self.reward_scalar
                    done = True
            else:  # "simple"
                if is_desired:
                    reward = self.reward_scalar

            # only allow max number of undesired steps in a row
            if is_desired:
                self.undesired_steps = 0
            else:
                self.undesired_steps += 1
            if self.max_undesired_steps is not None and \
               self.undesired_steps >= self.max_undesired_steps:
                done = True
        else:
            # continuous
            self.state[0] += action
            reward = -np.abs(self.desired - self.state[0])

        return copy.deepcopy(self.state), reward, done, {}

    def render(self, **kwargs):
        pass
