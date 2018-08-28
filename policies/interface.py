import numpy as np
from gym import Space
from gym.spaces import Discrete


class Interface(object):
    def __init__(self, space=None, deterministic=True):
        if space is not None:
            # get shape and discreteness for interface
            if isinstance(space, Discrete):
                self.shape, self.size, self.discrete = (space.n, ), space.n, True
            elif isinstance(space, Space):
                self.shape, self.size, self.discrete = space.shape, np.prod(space.shape), False
            else:
                print(f"Invalid interface space: {space}")
            self.dtype = space.dtype
        self.deterministic = deterministic

    def __str__(self):
        discreteness = "discrete" if self.discrete else "continuous"
        deterministicness = "deterministic" if self.deterministic else "stochastic"
        return f"Interface({self.shape} [{self.size}], {discreteness}, {deterministicness})"

    def encode(self, value):
        # handle discrete values
        if self.discrete:
            discretized = np.zeros(self.shape)
            discretized[value] = 1
            return discretized
        return value

    def decode(self, value):
        # handle discrete values
        if self.discrete:
            if self.deterministic:
                return np.argmax(value)
            else:
                return np.random.choice(range(len(value)), p=value)
        return value
