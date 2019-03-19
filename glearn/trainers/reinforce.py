import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class ReinforceTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, **kwargs):
        self.gamma = gamma

        super().__init__(config, **kwargs)

    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # get log prob of action
            with tf.name_scope("loss"):
                policy_distribution = self.policy.network.get_distribution_layer()
                actions = self.get_feed("Y")
                # log_prob_actions = policy_distribution.log_prob(actions)
                neg_log_prob_actions = policy_distribution.neg_log_prob(actions)

                discount_rewards = self.create_feed("discount_rewards", query, (None, 1))
                policy_loss = tf.reduce_mean(neg_log_prob_actions * discount_rewards)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize policy loss
            self.policy.optimize_loss(policy_loss, name=query)

    # def process_transition(self, transition):
    #     # store log prob of action
    #     transition.info["log_prob_action"] = self.latest_results["log_prob_action"]

    def calculate_discount_rewards(self, rewards):
        # gather discounted rewards
        trajectory_length = len(rewards)
        reward = 0
        discount_rewards = np.zeros(trajectory_length)
        for i in reversed(range(trajectory_length)):
            reward = rewards[i] + self.gamma * reward
            discount_rewards[i] = reward

        # normalize
        std = np.std(discount_rewards)
        mean = np.mean(discount_rewards)
        discount_rewards = (discount_rewards - mean) / std
        discount_rewards = np.expand_dims(discount_rewards, -1)
        return discount_rewards

    # def prepare_feeds(self, queries, feed_map):
    #     if intersects(["policy_optimize", "evaluate"], queries):
    #         # build value feed map with rewards
    #         if self.batch is not None:
    #             # compute discounted rewards
    #             batch = self.batch
    #             discount_rewards = self.calculate_discount_rewards(batch.rewards)
    #             feed_map["discount_rewards"] = discount_rewards
    #         else:
    #             shape = np.shape(feed_map["X"])[:-1] + (1,)
    #             feed_map["discount_rewards"] = np.zeros(shape)

    #     return super().prepare_feeds(queries, feed_map)

    def optimize(self, batch, feed_map):
        # compute discounted rewards
        batch = self.batch
        discount_rewards = self.calculate_discount_rewards(batch.rewards)
        feed_map["discount_rewards"] = discount_rewards

        # run desired queries
        return self.run(["policy_optimize"], feed_map)
