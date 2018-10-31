import numpy as np
from gym import Space
from gym.spaces import Discrete


class Interface(object):
    def __init__(self, space, deterministic=True):
        # get shape and discreteness for interface
        if isinstance(space, Discrete):
            self.shape, self.size, self.discrete = (space.n, ), space.n, True
        elif isinstance(space, Space):
            self.shape, self.size, self.discrete = space.shape, np.prod(space.shape), False
        else:
            print(f"Invalid interface space: {space}")
        self.space = space
        self.dtype = space.dtype
        self.deterministic = deterministic

    def __str__(self):
        properties = [
            f"{self.shape} [{self.size}]",
            "discrete" if self.discrete else "continuous",
            self.dtype.name,
            "deterministic" if self.deterministic else "stochastic",
        ]
        return f"Interface({', '.join(properties)})"

    def sample(self):
        if self.discrete:
            result = np.zeros(self.size)
            result[np.random.randint(0, self.size)] = 1
            return result
        else:
            return self.space.sample()

    def encode(self, value):
        result = value

        # handle discrete values
        if self.discrete:
            discretized = np.zeros(self.shape)
            discretized[value] = 1
            result = discretized

        return result

    def decode(self, value):
        result = value

        # handle discrete values
        if self.discrete:
            if self.deterministic:
                result = np.argmax(value)
            else:
                result = np.random.choice(range(len(value)), p=value)

        return result
