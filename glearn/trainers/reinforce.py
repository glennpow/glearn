import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class ReinforceTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, **kwargs):
        self.gamma = gamma

        self._zero_reward_warning = False

        super().__init__(config, **kwargs)

    def on_policy(self):
        return True  # False  # TODO - could be either?

    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # build loss based on negative log prob of actions and discount rewards
            with tf.name_scope("loss"):
                policy_distribution = self.policy.network.get_distribution_layer()
                actions = self.get_feed("Y")
                neg_logp = policy_distribution.neg_log_prob(actions)

                # HACK
                # self.add_metric("neg_logp", tf.reduce_mean(neg_logp), query=query)

                discount_rewards = self.create_feed("discount_rewards", query, (None, 1))

                # FIXME
                # policy_loss = tf.reduce_mean(neg_logp * discount_rewards)
                policy_loss = tf.reduce_mean(tf.reduce_sum(neg_logp * discount_rewards, -1))
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize policy loss
            self.policy.optimize_loss(policy_loss, name=query)

    def calculate_discount_rewards(self, rewards):
        # gather discounted rewards
        trajectory_length = len(rewards)
        reward = 0
        discount_rewards = np.zeros(trajectory_length, dtype=np.float32)
        for i in reversed(range(trajectory_length)):
            reward = rewards[i] * self.gamma + reward
            discount_rewards[i] = reward

        # normalize and reshape
        mean = np.mean(discount_rewards)
        discount_rewards = discount_rewards - mean
        std = np.std(discount_rewards)
        if std > 0:
            discount_rewards /= std
        discount_rewards = np.expand_dims(discount_rewards, -1).tolist()
        return discount_rewards

    def reset(self, mode="train", **kwargs):
        if mode == "test":
            self._zero_reward_warning = False

        return super().reset(**kwargs)

    def process_episode(self, episode):
        # compute discounted rewards
        discount_rewards = self.calculate_discount_rewards(episode["reward"])

        # ignore zero-reward episodes
        if np.allclose(discount_rewards, np.zeros_like(discount_rewards)):
            if not self._zero_reward_warning:
                self.warning("Ignoring episode(s) with zero rewards!")
                self._zero_reward_warning = True
            return False

        # HACK
        self.summary.add_simple_value("discount_episode_reward", discount_rewards[0][0])

        episode["discount_rewards"] = discount_rewards
        return True

    def optimize(self, batch):
        feed_map = batch.prepare_feeds()

        # feed discounted rewards
        feed_map["discount_rewards"] = batch["discount_rewards"]

        # run desired queries
        return self.run(["policy_optimize"], feed_map)
