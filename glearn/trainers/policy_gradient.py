import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class PolicyGradientTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, V=None, **kwargs):
        self.gamma = gamma
        self.V_definition = V

        super().__init__(config, **kwargs)

    def on_policy(self):
        return True  # False  # TODO - could be either for REINFORCE?

    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            state = self.get_feed("X")
            actions = self.get_feed("Y")

            if self.V_definition:
                V_network = self.build_network("V", self.V_definition, state)

            # build loss based on negative log prob of actions and discount rewards
            with tf.name_scope("loss"):
                # build -log(P(y))
                policy_distribution = self.policy.network.get_distribution_layer()
                neg_log_prob = policy_distribution.neg_log_prob(actions)

                # feed for discount rewards
                discount_rewards = self.create_feed("discount_rewards", query, (None,))

                if self.V_definition:
                    # subtract optional baseline (build advantage)
                    advantage = discount_rewards - V_network.outputs
                    V_loss = tf.reduce_mean(tf.square(advantage))

                    V_network.optimize_loss(V_loss, name="V_optimize")
                else:
                    # don't use baseline
                    # advantage = discount_rewards

                    # HACK - simple baseline
                    baseline = tf.reduce_sum(state)
                    advantage = discount_rewards - baseline

                policy_loss = tf.reduce_mean(neg_log_prob * advantage)
            self.add_metric("policy_loss", policy_loss, query=query)
            if self.V_definition:
                self.add_metric("V_loss", V_loss, query="V_optimize")

            # minimize policy loss
            self.policy.optimize_loss(policy_loss, name=query)

            # DEBUG ====================
            average_neg_log_prob = tf.reduce_mean(neg_log_prob)
            self.summary.add_scalar("neg_log_prob", average_neg_log_prob, query=query)
            entropy = policy_distribution.entropy()
            average_entropy = tf.reduce_mean(entropy)
            self.summary.add_scalar("entropy", average_entropy, query=query)
            # self.summary.add_histogram("entropy", entropy, query=query)
            confidence = policy_distribution.prob(actions)
            self.summary.add_histogram("confidence", confidence, query=query)
            average_discount_rewards = tf.reduce_mean(discount_rewards)
            self.summary.add_scalar("discount_rewards", average_discount_rewards, query=query)

            if self.output.discrete:
                probs = policy_distribution.probs
                probs = tf.transpose(probs)
                for i in range(probs.shape[0]):
                    self.summary.add_histogram(f"prob_{i}", probs[i], query=query)
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

    def process_episode(self, episode):
        if not super().process_episode(episode):
            return False

        # compute discounted rewards
        discount_rewards = self.calculate_discount_rewards(episode["reward"])
        episode["discount_rewards"] = discount_rewards

        return True

    def optimize(self, batch):
        fetches = ["policy_optimize"]
        feed_map = batch.prepare_feeds()

        # feed discounted rewards
        feed_map["discount_rewards"] = batch["discount_rewards"]

        # optimize baseline
        if self.V_definition:
            fetches.append("V_optimize")

        # run desired queries
        return self.run(fetches, feed_map)
