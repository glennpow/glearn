import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class ReinforceTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, **kwargs):
        self.gamma = gamma

        self._zero_reward_warning = False
        self._discount_episode_rewards = []

        super().__init__(config, **kwargs)

    def on_policy(self):
        return True  # False  # TODO - could be either for REINFORCE?

    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # build loss based on negative log prob of actions and discount rewards
            with tf.name_scope("loss"):
                actions = self.get_feed("Y")
                
                policy_distribution = self.policy.network.get_distribution_layer()
                neg_log_prob = policy_distribution.neg_log_prob(actions)
                average_neg_log_prob = tf.reduce_mean(neg_log_prob)

                discount_rewards = self.create_feed("discount_rewards", query, (None,))
                average_discount_rewards = tf.reduce_mean(discount_rewards)

                # policy_loss = tf.reduce_mean(average_neg_log_prob * average_discount_rewards)
                policy_loss = tf.reduce_mean(neg_log_prob * discount_rewards)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize policy loss
            self.policy.optimize_loss(policy_loss, name=query)

            # DEBUG ====================
            self.summary.add_scalar("neg_log_prob", average_neg_log_prob, query=query)

            if self.output.discrete:
                probs = policy_distribution.probs
                probs = tf.transpose(probs)
                for i in range(probs.shape[0]):
                    self.summary.add_histogram(f"prob_{i}", probs[i], query=query)

            confidence = policy_distribution.prob(actions)
            self.summary.add_histogram("confidence", confidence, query=query)
            self.summary.add_scalar("discount_rewards", average_discount_rewards, query=query)
            # ==========================

    def calculate_discount_rewards(self, rewards):
        # gather discounted rewards
        trajectory_length = len(rewards)
        reward = 0
        discount_rewards = np.zeros(trajectory_length, dtype=np.float32)
        for i in reversed(range(trajectory_length)):
            reward = rewards[i] + self.gamma * reward
            discount_rewards[i] = reward

        # normalize and reshape
        mean = np.mean(discount_rewards)
        discount_rewards = discount_rewards - mean
        std = np.std(discount_rewards)
        if std > 0:
            discount_rewards /= std
        return discount_rewards

    def reset(self, mode="train", **kwargs):
        if mode == "test":
            self._zero_reward_warning = False

        return super().reset(**kwargs)

    def process_episode(self, episode):
        # compute discounted rewards
        discount_rewards = self.calculate_discount_rewards(episode["reward"])
        episode["discount_rewards"] = discount_rewards

        # ignore zero-reward episodes
        if np.count_nonzero(discount_rewards) == 0:
            if not self._zero_reward_warning:
                self.warning("Ignoring episode(s) with zero rewards!")
                self._zero_reward_warning = True
            return False

        # track average discounted episode reward
        # self._discount_episode_rewards.append(discount_rewards[0])

        return True

    def optimize(self, batch):
        feed_map = batch.prepare_feeds()

        # summary of discounted episode rewards
        # average_episode_reward = np.mean(self._discount_episode_rewards)
        # self._discount_episode_rewards = []
        # self.summary.add_simple_value("discount_episode_reward", average_episode_reward)

        # feed discounted rewards
        feed_map["discount_rewards"] = batch["discount_rewards"]

        # run desired queries
        return self.run(["policy_optimize"], feed_map)
