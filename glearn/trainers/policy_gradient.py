import numpy as np
import tensorflow as tf
from glearn.trainers.reinforcement import ReinforcementTrainer


class PolicyGradientTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, V=None, V_coef=None, normalize_advantage=True,
                 ent_coef=None, simple_baseline=False, **kwargs):
        self.gamma = gamma
        self.V_definition = V
        self.V_coef = V_coef
        self.ent_coef = ent_coef
        self.normalize_advantage = normalize_advantage
        self.simple_baseline = simple_baseline

        super().__init__(config, **kwargs)

        # policy gradient only works for RL
        assert self.has_env

    def build_trainer(self):
        # get the inputs
        state = self.get_feed("X")
        action = self.get_feed("Y")

        # optimize policy
        self.optimize_policy(state, action)

    def optimize_policy(self, state, action):
        self.policy_network = self.policy.network

        query = "policy_optimize"
        with tf.name_scope(query):
            # build expected return
            expected_return = self.build_expected_return(state, action, query=query)

            # build loss: -log(P(y)) * (discount rewards - optional baseline)
            with tf.name_scope("loss"):
                # build -log(P(y))
                policy_distribution = self.policy.network.get_distribution_layer()
                neg_log_prob = policy_distribution.neg_log_prob(action)

                # build policy loss
                policy_loss = tf.reduce_mean(neg_log_prob * expected_return)
                self.policy_network.add_loss(policy_loss)

                # build entropy loss
                if self.ent_coef:
                    entropy = policy_distribution.entropy()
                    self.policy_network.add_loss(self.ent_coef * entropy)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize total policy loss
            total_policy_loss = self.policy_network.build_total_loss()
            self.policy.optimize_loss(total_policy_loss, name=query)

            # ==================== DEBUG ====================
            average_neg_log_prob = tf.reduce_mean(neg_log_prob)
            self.summary.add_scalar("neg_log_prob", average_neg_log_prob, query=query)
            if self.ent_coef:
                self.summary.add_scalar("entropy", tf.reduce_mean(entropy), query=query)
                # self.summary.add_histogram("entropy", entropy, query=query)
            confidence = policy_distribution.prob(action)
            self.summary.add_histogram("confidence", confidence, query=query)
            # average_discount_rewards = tf.reduce_mean(discount_rewards)
            # self.summary.add_scalar("discount_rewards", average_discount_rewards, query=query)

            if self.output.discrete:
                probs = policy_distribution.probs
                probs = tf.transpose(probs)
                for i in range(probs.shape[0]):
                    self.summary.add_histogram(f"prob_{i}", probs[i], query=query)
            # ===============================================

    def build_expected_return(self, state, action, query=None):
        # feed for discount rewards
        target = self.build_target(state, action, query=query)

        # subtract optional baseline
        baseline = self.build_baseline(state)

        if baseline:
            advantage = self.build_advantage(target, baseline, normalize=self.normalize_advantage,
                                             query=query)

            # optimize optional baseline
            self.optimize_baseline()

            return advantage
        else:
            return target

    def build_target(self, state, action, query=None):
        # feed for discount rewards
        return self.create_feed("discount_rewards", shape=(None,), queries=query)

    def build_baseline(self, state):
        # build optional baseline
        has_baseline = self.V_definition or self.simple_baseline
        if has_baseline:
            with tf.name_scope("baseline"):
                if self.V_definition:
                    # V-network baseline
                    V_network = self.build_V(state)
                    # self.add_metric("V", V_network.outputs)
                    return V_network.outputs
                elif self.simple_baseline:
                    # Simple baseline  (FIXME - One of Woj's suggestions.  ask him.)
                    return tf.reduce_sum(state)  # TODO - some function of non-params
        return None

    def optimize_baseline(self):
        # optimize baseline V-network
        if self.V_definition:
            with tf.name_scope("baseline/"):
                V_loss = self.optimize_V()

                self.policy_network.add_loss(V_loss * self.V_coef)

    def calculate_discount_rewards(self, rewards):
        # gather discounted rewards
        trajectory_length = len(rewards)
        reward = 0
        discount_rewards = np.zeros(trajectory_length, dtype=np.float32)
        for i in reversed(range(trajectory_length)):
            reward = rewards[i] + self.gamma * reward
            discount_rewards[i] = reward

        # normalize
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
        episode["discount_rewards"] = self.calculate_discount_rewards(episode["reward"])

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
