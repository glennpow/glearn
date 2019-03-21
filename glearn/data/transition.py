import numpy as np
from glearn.data.batch import Batch


class Transition(object):
    def __init__(self, state, action, reward, next_state, done, info):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info


class TransitionBatch(Batch):
    def __init__(self, transitions, mode="train"):
        super().__init__(mode=mode)

        self.inputs = [t.state for t in transitions]
        self.outputs = [t.action for t in transitions]
        self.rewards = np.array([t.reward for t in transitions])
        self.next_states = [t.next_state for t in transitions]
        self.dones = np.array([t.done for t in transitions])

        self.transitions = transitions
