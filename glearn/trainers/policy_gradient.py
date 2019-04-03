import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class PolicyGradientTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, V=None, simple_baseline=False, **kwargs):
        self.gamma = gamma
        self.V_definition = V
        self.simple_baseline = simple_baseline

        super().__init__(config, **kwargs)

    def on_policy(self):
        return True  # False  # TODO - could be either?

    def build_trainer(self):
        # get the goods
        state = self.get_feed("X")
        actions = self.get_feed("Y")

        # build optional baseline
        baseline = self.build_baseline(state)

        query = "policy_optimize"
        with tf.name_scope(query):
            # build loss: -log(P(y)) * (discount rewards - optional baseline)
            with tf.name_scope("loss"):
                # build -log(P(y))
                policy_distribution = self.policy.network.get_distribution_layer()
                neg_log_prob = policy_distribution.neg_log_prob(actions)

                # feed for discount rewards
                discount_rewards = self.create_feed("discount_rewards", query, (None,))

                # subtract optional baseline (build advantage)
                if baseline:
                    advantage = discount_rewards - baseline
                else:
                    advantage = discount_rewards

                # build policy loss
                policy_loss = tf.reduce_mean(neg_log_prob * advantage)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize policy loss
            self.policy.optimize_loss(policy_loss, name=query)

            # ==================== DEBUG ====================
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
            # ===============================================

        # optimize optional baseline
        self.optimize_baseline(advantage)

    def build_baseline(self, state):
        # build optional baseline
        has_baseline = self.V_definition or self.simple_baseline
        if has_baseline:
            with tf.name_scope("baseline"):
                if self.V_definition:
                    # V-network baseline
                    V_network = self.build_network("V", self.V_definition, state)
                    return V_network.outputs
                elif self.simple_baseline:
                    # Simple baseline  (FIXME - One of Woj's suggestions.  ask him.)
                    return tf.reduce_sum(state)  # TODO - some function of non-params
        return None

    def optimize_baseline(self, advantage):
        # optimize baseline V-network
        if self.V_definition:
            with tf.name_scope("baseline/"):
                query = "V_optimize"
                with tf.name_scope(query):
                    # MSE V-loss
                    with tf.name_scope("loss"):
                        V_loss = tf.reduce_mean(tf.square(advantage))
                    self.add_metric("V_loss", V_loss, query=query)

                    # minimize V-loss
                    self.networks["X"].optimize_loss(V_loss, name=query)

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
