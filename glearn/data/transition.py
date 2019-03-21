from .batch import Batch


class Transition(object):
    def __init__(self, state, action, reward, next_state, done, info, timestamp):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info
        self.timestamp = timestamp


class TransitionBatch(Batch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reward = 0

    def add_transition(self, transition):
        # accumulate episode reward
        self.reward += transition.reward

        # build sample
        sample = {
            "state": transition.state,
            "action": transition.action,
            "reward": transition.reward,
            "next_state": transition.next_state,
            "done": transition.done,
            "timestamp": transition.timestamp,
        }
        if transition.info:
            for key, value in transition.info.items():
                sample[key] = value

        self.add_sample(sample)

    def transition_count(self):
        return len(self)

    def prepare_feeds(self):
        return {
            "X": self.samples["state"],
            "Y": self.samples["action"],
        }
