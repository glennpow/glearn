import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from .maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, gen_mode=None,
                 obs_coord=True, obs_walls=True):
        self.viewer = None

        if maze_file:
            self.viewer = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                     maze_file_path=maze_file,
                                     screen_size=(640, 640))
        elif maze_size:
            if gen_mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size) / 3))
            else:
                has_loops = False
                num_portals = 0

            self.viewer = MazeView2D(maze_name=f"OpenAI Gym - Maze ({maze_size})",
                                     maze_size=maze_size, screen_size=(640, 640),
                                     has_loops=has_loops, num_portals=num_portals)
        else:
            raise AttributeError("One must supply either a maze_file path (str) "
                                 "or the maze_size (tuple of length 2)")

        self.maze_size = self.viewer.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2 * len(self.maze_size))

        self.obs_coord = obs_coord
        if self.obs_coord:
            # observation contains the x, y coordinate of the grid
            low = np.zeros(len(self.maze_size), dtype=int)
            high = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        else:
            low = []
            high = []
        self.obs_walls = obs_walls
        if self.obs_walls:
            # observation contains 4 walls
            low = np.append(low, np.zeros(4, dtype=int))
            high = np.append(high, np.ones(4, dtype=int))
        self.observation_space = spaces.Box(low, high)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def __del__(self):
        self.viewer.quit_game()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.viewer.move_robot(self.ACTION[action])

        if np.array_equal(self.viewer.robot, self.viewer.goal):
            reward = 1
            done = True
        else:
            reward = -0.1 / (self.maze_size[0] * self.maze_size[1])
            done = False

        if self.obs_coord:
            self.state = self.viewer.robot
        else:
            self.state = []
        if self.obs_walls:
            walls = [1 if self.viewer.can_move_robot(dir) else 0 for dir in self.ACTION]
            self.state = np.append(self.state, walls)

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.viewer.reset_robot()
        self.state = np.zeros(self.observation_space.shape)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.viewer.game_over

    def render(self, mode="human", close=False):
        if close:
            self.viewer.quit_game()

        return self.viewer.render(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy")


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5))


class MazeEnvSample10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy")


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10))


class MazeEnvSample3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy")


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3))


class MazeEnvSample100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy")


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100))


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus")


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus")


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus")
