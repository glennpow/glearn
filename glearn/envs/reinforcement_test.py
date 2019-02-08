import numpy as np
import gym
from gym import spaces


class ReinforcementTestEnv(gym.Env):
    def __init__(self, action_size=4, desired=1, mode=None, reward_multiple=1,
                 max_undesired_steps=None):
        self.action_size = action_size
        self.desired = desired
        self.mode = mode
        self.reward_multiple = reward_multiple
        self.max_undesired_steps = max_undesired_steps

        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(np.zeros(1, np.int32), np.ones(1, np.int32))
        self.state = np.zeros(self.observation_space.shape)

    def reset(self):
        self.undesired_steps = 0
        return self.state

    def step(self, action):
        predicted = np.argmax(action)
        reward = 0
        done = False
        is_desired = predicted == self.desired

        if self.mode == "sparse":
            if is_desired:
                reward = self.reward_multiple
                done = True
        elif self.mode == "decaying":
            reward = -1
            if is_desired:
                reward = self.action_size * self.reward_multiple
                done = True
        else:  # "simple"
            if is_desired:
                reward = self.reward_multiple

        if not is_desired:
            self.undesired_steps += 1
        if self.max_undesired_steps is not None and \
           self.undesired_steps >= self.max_undesired_steps:
            done = True

        return self.state, reward, done, {}

    def render(self, **kwargs):
        pass
