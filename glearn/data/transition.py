import time
import numpy as np
from .buffer import Buffer


class Transition(object):
    def __init__(self, step, timestamp, state, action, reward, next_state, done, info):
        self.step = step
        self.timestamp = timestamp
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info


class TransitionBuffer(Buffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reward = 0

    def clear(self):
        super().clear()

        self.reward = 0

    def add_transition(self, transition):
        # accumulate episode reward
        self.reward += transition.reward

        # build sample
        sample = {
            "step": transition.step,
            "timestamp": transition.timestamp,
            "state": transition.state,
            "action": transition.action,
            "reward": transition.reward,
            "next_state": transition.next_state,
            "done": transition.done,
        }
        if transition.info:
            for key, value in transition.info.items():
                sample[key] = value

        self.add_sample(sample)

    def get_ages(self):
        if self.empty():
            return np.array([])
        t = time.time()
        return np.array([t - ts if ts > 0 else 0 for ts in self.samples["timestamp"]])

    def transition_count(self):
        return self.sample_count()

    def get_feeds(self):
        return {
            "X": self.samples["state"],
            "Y": self.samples["action"],
        }
