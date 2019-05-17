import numpy as np
from gym import Space
from gym.spaces import Discrete


class Interface(object):
    def __init__(self, space, deterministic=True):
        # get shape, type and discreteness for interface
        if isinstance(space, Discrete):
            self.shape, self.size, self.discrete = space.shape, space.n, True
            self.dtype = np.dtype(np.int32)
        elif isinstance(space, Space):
            self.shape, self.size = space.shape, np.prod(space.shape)
            self.discrete = np.issubdtype(space.dtype, np.integer)
            self.dtype = space.dtype
        else:
            print(f"Invalid interface space: {space}")
        self.space = space
        self.deterministic = deterministic

    def __str__(self):
        properties = [
            f"{self.shape} [{self.size}]",
            "discrete" if self.discrete else "continuous",
            self.dtype.name,
            "deterministic" if self.deterministic else "stochastic",
        ]
        return f"Interface({', '.join(properties)})"

    @property
    def stochastic(self):
        return not self.deterministic

    @property
    def continuous(self):
        return not self.discrete

    def sample(self):
        if self.discrete:
            result = np.random.randint(0, self.size)
            return result
        else:
            return self.space.sample()
