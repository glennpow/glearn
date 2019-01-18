import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.utils.collections import intersects


class TDTrainer(Trainer):
    # option to precompute this at rollout time
    precompute_discounted_rewards = False

    def __init__(self, config, policy, gamma=0.95, normalize_advantage=False, **kwargs):
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        super().__init__(config, policy, **kwargs)

    def build_value_loss(self, value):
        policy = self.policy

        with tf.name_scope("loss"):
            # calculate advantage (td_error), using discounted rewards (td_target)
            discounted_reward = policy.create_feed("discounted_reward",
                                                   ["policy_optimize", "value_optimize",
                                                    "evaluate"], (None, 1))
            advantage = discounted_reward - value
            if self.normalize_advantage:  # aborghi implementation
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy.set_fetch("advantage", advantage)

            # value loss minimizes squared advantage
            value_loss = tf.reduce_mean(tf.square(advantage))
            policy.set_fetch("value_loss", value_loss, "evaluate")

            # averages
            discounted_reward = tf.reduce_mean(discounted_reward)
            advantage = tf.reduce_mean(advantage)

        # summaries
        self.summary.add_scalar("value_loss", value_loss, "evaluate")
        self.summary.add_scalar("discount_reward", discounted_reward, "evaluate")
        self.summary.add_scalar("advantage", advantage, "evaluate")

        return value_loss

    def compute_value(self, state):
        feed_map = {"X": [state]}
        return self.fetch("value", feed_map, squeeze=True)

    def compute_values(self, states):
        feed_map = {"X": states}
        return self.fetch("value", feed_map)

    def prepare_feeds(self, families, feed_map):
        if intersects(["policy_optimize", "value_optimize", "evaluate"], families):
            # build value feed map with rewards
            if self.batch is not None:
                transitions = self.batch.info["transitions"]
                if self.precompute_discounted_rewards:
                    # use pre-computed discounted rewards
                    discounted_rewards = [t.discounted_reward for t in transitions]
                    feed_map["discounted_reward"] = np.array(discounted_rewards)[:, np.newaxis]
                else:
                    # compute discounted rewards
                    new_states = [t.new_observation for t in transitions]
                    rewards = np.array([t.reward for t in transitions])[:, np.newaxis]
                    new_values = self.compute_values(new_states)
                    discounted_rewards = np.add(rewards, self.gamma * new_values)
                    feed_map["discounted_reward"] = discounted_rewards
            else:
                shape = np.shape(feed_map["X"])[:-1] + (1,)
                feed_map["discounted_reward"] = np.zeros(shape)

        return super().prepare_feeds(families, feed_map)

    def process_transition(self, transition):
        if self.precompute_discounted_rewards:
            if transition.done:
                # done, so use complete reward
                transition.discounted_reward = transition.reward
            else:
                # compute the gamma discounted reward (td_target) using estim value of next state
                new_value = self.compute_value(transition.new_observation)
                transition.discounted_reward = transition.reward + self.gamma * new_value
