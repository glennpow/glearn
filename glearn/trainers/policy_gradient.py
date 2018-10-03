import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def __init__(self, config, policy, epsilon=0, keep_prob=1, **kwargs):
        # get basic params
        self.epsilon = epsilon
        self.keep_prob = keep_prob

        super().__init__(config, policy, **kwargs)

    def init_optimizer(self):
        # minimize loss
        with tf.name_scope('optimize'):
            optimize = self.optimize_loss()
            self.policy.set_fetch("optimize", optimize)

        # get accuracy summary from policy
        accuracy = self.policy.get_fetch("accuracy", "evaluate")
        if accuracy is not None:
            self.summary.add_scalar("accuracy", accuracy, "evaluate")

    def prepare_feeds(self, graphs, feed_map):
        feed_map = super().prepare_feeds(graphs, feed_map)

        if "optimize" in graphs:
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1
        return feed_map

    def action(self):
        # decaying epsilon-greedy
        # FIXME - should this be per epoch/episode instead of iteration?
        epsilon = self.epsilon
        if isinstance(epsilon, list):
            t = min(1, self.global_step / epsilon[2])
            epsilon = t * (epsilon[1] - epsilon[0]) + epsilon[0]

        # get action
        if np.random.random() < epsilon:
            # choose epsilon-greedy random action  (TODO - could implement this in tf)
            return self.output.sample()
        else:
            # choose optimal policy action
            return super().action()
