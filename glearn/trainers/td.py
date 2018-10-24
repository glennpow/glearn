import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.utils.collections import intersects


class TDTrainer(Trainer):
    def __init__(self, config, policy, gamma=0.95, normalize_advantage=False, **kwargs):
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage  # TODO - try this out?

        super().__init__(config, policy, **kwargs)

    def init_value_loss(self, value):
        policy = self.policy

        # build advantage and value optimization
        with tf.name_scope('td'):
            # calculate advantage, using discounted rewards
            # discounted_reward = reward + gamma * value  # this happens out of graph now
            discounted_reward = policy.create_feed("discounted_reward", ["advantage"], (None, 1))
            advantage = discounted_reward - value
            if self.normalize_advantage:  # aborghi implementation
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy.set_fetch("advantage", advantage)

            # value loss minimizes squared advantage
            value_loss = tf.reduce_mean(tf.square(advantage))
            policy.set_fetch("value_loss", value_loss, "evaluate")

        # summaries
        self.summary.add_scalar("value_loss", value_loss, "evaluate")
        self.summary.add_scalar("advantage", tf.reduce_mean(advantage), "evaluate")
        self.summary.add_scalar("advantage_abs", tf.reduce_mean(tf.abs(advantage)), "evaluate")
        return value_loss

    def prepare_feeds(self, graphs, feed_map):
        if intersects(["advantage", "value_optimize"], graphs):
            # build value feed map with rewards
            if self.batch is not None:
                reward = [e.discounted_reward for e in self.batch.info["transitions"]]
                feed_map["discounted_reward"] = np.array(reward)[:, np.newaxis]
            else:
                shape = np.shape(feed_map["X"])[:-1] + (1,)
                feed_map["discounted_reward"] = np.zeros(shape)

        return super().prepare_feeds(graphs, feed_map)

    def process_transition(self, transition):
        # get value value and apply discount gamma here, during rollouts
        # TODO - could get this along WITH 'predict' fetch before?
        feed_map = {"X": [transition.observation]}

        value = self.fetch("value", feed_map, squeeze=True)
        transition.discounted_reward = transition.reward + self.gamma * value
