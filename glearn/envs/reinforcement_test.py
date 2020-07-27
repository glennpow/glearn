import numpy as np
import gym
from gym import spaces


# DEPRECATE


class ReinforcementTestEnv(gym.Env):
    def __init__(self, action_size=4, desired=1, desired_sequence=False, desired_terminates=False,
                 discrete=False, sparse_reward=False, reward_scalar=1, max_undesired_steps=None):
        self.action_size = action_size
        self.desired = desired
        self.desired_sequence = desired_sequence
        self.desired_terminates = desired_terminates
        self.discrete = discrete
        self.sparse_reward = sparse_reward
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

    def reset(self):
        self.undesired_steps = 0
        self.desired_index = 0
        self.state = np.zeros(self.obs_shape, self.obs_dtype)
        if isinstance(self.desired, list):
            self.state[0] = self.desired[self.desired_index]
        return np.copy(self.state)

    def step(self, action):
        reward = 0
        done = False

        if self.discrete:
            if self.desired is None:
                # random desired shown in state
                is_desired = action == self.state[0]
                self.state[0] = np.random.randint(0, self.action_size)
            else:
                # check if desired action has been performed
                if isinstance(self.desired, list):
                    if self.desired_sequence:
                        # must sent desired in sequence
                        is_next_desired = action == self.desired[self.desired_index]
                        if is_next_desired:
                            self.desired_index = (self.desired_index + 1) % len(self.desired)
                            if self.sparse_reward:
                                is_desired = self.desired_index == 0
                            else:
                                is_desired = is_next_desired
                        else:
                            is_desired = False
                            # self.desired_index = 0  # this will only learn first digit

                        self.state[0] = self.desired[self.desired_index]
                    else:
                        is_desired = action in self.desired
                        self.state[0] = action
                else:
                    is_desired = action == self.desired
                    self.state[0] = action

            # reward
            if not self.sparse_reward:
                reward = -1
            if is_desired:
                reward = self.reward_scalar
                if self.desired_terminates:
                    done = True

            # only allow max number of undesired steps per episode
            if self.max_undesired_steps is not None:
                if not is_desired:
                    self.undesired_steps += 1
                if self.undesired_steps >= self.max_undesired_steps:
                    done = True
        else:
            # continuous
            self.state[0] += action
            reward = -np.abs(self.desired - self.state[0])

        # print(f"{action} => {self.state}, {reward}, {done}")
        return np.copy(self.state), reward, done, {}

    def render(self, **kwargs):
        pass
