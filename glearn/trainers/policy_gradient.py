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

        # build expected return
        expected_return = self.build_expected_return(state, action, query=query)

        with tf.variable_scope(query):
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
                    entropy = tf.reduce_mean(policy_distribution.entropy())
                    entropy_loss = self.ent_coef * -entropy
                    self.policy_network.add_loss(entropy_loss)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize total policy loss
            total_policy_loss = self.policy_network.build_total_loss()
            self.policy.optimize_loss(total_policy_loss, name=query)

            # ==================== DEBUG ====================
            average_neg_log_prob = tf.reduce_mean(neg_log_prob)
            self.summary.add_scalar("neg_log_prob", average_neg_log_prob, query=query)
            if self.ent_coef:
                self.summary.add_scalar("entropy", entropy, query=query)
                self.summary.add_scalar("entropy_loss", entropy_loss, query=query)
            confidence = policy_distribution.prob(action)
            self.summary.add_histogram("confidence", confidence, query=query)

            if self.output.discrete:
                probs = policy_distribution.probs
                probs = tf.transpose(probs)
                for i in range(probs.shape[0]):
                    self.summary.add_histogram(f"prob_{i}", probs[i], query=query)
            # ===============================================

    def build_expected_return(self, state, action, query=None):
        # feed for discount rewards
        target = self.build_target(state, action, query=query)

        # build with optional subtracted baseline
        expected_return = self.build_baseline_expected_return(state, target, query=query)

        if expected_return is None:
            expected_return = target

        return expected_return

    def build_target(self, state, action, query=None):
        # feed for discount rewards
        return self.create_feed("target", shape=(None,), queries=query)

    def get_baseline_scope(self):
        return "baseline"

    def build_baseline_expected_return(self, state, target, query=None):
        # build optional baseline
        has_baseline = self.V_definition or self.simple_baseline
        if has_baseline:
            with tf.variable_scope(self.get_baseline_scope()):
                if self.V_definition:
                    # V-network baseline
                    V_network = self.build_V(state)
                    baseline = V_network.outputs

                    # summary
                    self.add_metric("V", tf.reduce_mean(baseline), query=query)
                elif self.simple_baseline:
                    # Simple baseline  (FIXME - One of Woj's suggestions.  ask him.)
                    baseline = tf.reduce_sum(state)  # TODO - some function of non-params

                    # summary
                    self.add_metric("baseline", tf.reduce_mean(baseline), query=query)

                # build advantage by subtracting baseline from target
                advantage = self.build_advantage(target, baseline,
                                                 normalize=self.normalize_advantage, query=query)

                # optimize baseline V-network
                if self.V_definition:
                    V_loss = self.optimize_V(advantage)

                    if self.V_coef is not None:
                        V_loss *= self.V_coef

                    self.policy_network.add_loss(V_loss)

            return advantage
        return None

    def calculate_discount_rewards(self, episode):
        # gather discounted rewards
        rewards = episode["reward"]
        dones = episode["done"]
        trajectory_length = len(rewards)
        discount_reward = 0

        # bootstrap incomplete episodes with estimated value
        if self.V_definition:
            if dones[-1] == 0:
                # TODO - could try to keep last_V around from rollouts
                last_V = self.fetch_V(episode["state"][-1])
                discount_reward = last_V

        # discount reward for all transitions
        discount_rewards = np.zeros(trajectory_length, dtype=np.float32)
        for i in reversed(range(trajectory_length)):
            # FIXME - I don't think I need the (1 - dones[i]) term...
            discount_reward = rewards[i] + self.gamma * discount_reward * (1 - dones[i])
            discount_rewards[i] = discount_reward

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
        episode["target"] = self.calculate_discount_rewards(episode)

        return True

    def prepare_feeds(self, queries, feed_map):
        super().prepare_feeds(queries, feed_map)

        if self.is_optimize(queries):
            # feed discounted rewards
            feed_map["target"] = self.batch["target"]

    def get_optimize_query(self, batch):
        query = super().get_optimize_query(batch)

        # optimize baseline
        if self.V_definition:
            query.append("V_optimize")

        return query
