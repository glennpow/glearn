import numpy as np
from glearn.trainers.reinforcement import ReinforcementTrainer
from glearn.utils.collections import intersects


# TODO - deprecate


class TemporalDifferenceTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, **kwargs):
        self.gamma = gamma

        self.create_feed("td_target",
                         ["policy_optimize", "V_optimize", "evaluate"], (None, 1))

        super().__init__(config, **kwargs)

    def current_value_name(self):
        return "V"

    def compute_value(self, state):
        feed_map = {"X": [state]}
        return self.fetch(self.current_value_name(), feed_map, squeeze=True)

    def compute_values(self, states):
        feed_map = {"X": states}
        return self.fetch(self.current_value_name(), feed_map)

    def prepare_feeds(self, queries, feed_map):
        super().prepare_feeds(queries, feed_map)

        if intersects(["policy_optimize", "V_optimize", "evaluate"], queries):
            # build value feed map with rewards
            if self.batch is not None:
                # compute discounted rewards
                batch = self.batch
                new_values = self.compute_values(batch["next_state"])
                td_targets = np.add(batch["reward"], self.gamma * new_values * (1 - batch["done"]))
                feed_map["td_target"] = td_targets
            else:
                shape = np.shape(feed_map["X"])[:-1] + (1,)
                feed_map["td_target"] = np.zeros(shape)
