import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class ReinforceTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, **kwargs):
        self.gamma = gamma

        super().__init__(config, **kwargs)

    def on_policy(self):
        return False  # TODO - could be either?

    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # build loss based on negative log prob of actions and discount rewards
            with tf.name_scope("loss"):
                policy_distribution = self.policy.network.get_distribution_layer()
                actions = self.get_feed("Y")
                # neg_logp = -policy_distribution.log_prob(actions)
                neg_logp = policy_distribution.neg_log_prob(actions)
                # neg_logp = policy_distribution.cross_entropy(actions)

                discount_rewards = self.create_feed("discount_rewards", query, (None, 1))
                policy_loss = tf.reduce_mean(neg_logp * discount_rewards)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize policy loss
            self.policy.optimize_loss(policy_loss, name=query)

    def calculate_discount_rewards(self, rewards):
        # gather discounted rewards
        trajectory_length = len(rewards)
        reward = 0
        discount_rewards = np.zeros(trajectory_length)
        for i in reversed(range(trajectory_length)):
            reward = rewards[i] + self.gamma * reward
            discount_rewards[i] = reward

        # normalize and reshape
        std = np.std(discount_rewards)
        mean = np.mean(discount_rewards)
        discount_rewards = (discount_rewards - mean) / std
        discount_rewards = np.expand_dims(discount_rewards, -1)
        return discount_rewards

    def optimize(self, batch):
        feed_map = batch.prepare_feeds()

        # compute discounted rewards
        # batch = self.batch
        discount_rewards = self.calculate_discount_rewards(batch["reward"])  # FIXME - do this in process_transition
        feed_map["discount_rewards"] = discount_rewards

        # run desired queries
        return self.run(["policy_optimize"], feed_map)
