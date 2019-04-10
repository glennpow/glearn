import numpy as np
import tensorflow as tf
from .reinforcement import ReinforcementTrainer


class PolicyGradientTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, V=None, V_coef=None, normalize_advantage=True,
                 ent_coef=None, gae_lambda=None, simple_baseline=False, **kwargs):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.V_definition = V
        self.V_coef = V_coef
        self.ent_coef = ent_coef
        self.normalize_advantage = normalize_advantage
        self.simple_baseline = simple_baseline

        self.V_bootstrap = False

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

        # feeds for targets
        advantage = self.create_feed("advantage", shape=(None,), query=query)

        # build optional baseline
        self.build_baseline(state)

        with tf.variable_scope(query):
            # build policy loss
            with tf.name_scope("loss"):
                policy_distribution = self.policy.network.get_distribution_layer()

                # # build policy loss
                policy_loss = self.build_policy_loss(action, advantage, query=query)
                self.policy_network.add_loss(policy_loss)

                # build entropy loss
                if self.ent_coef:
                    entropy = tf.reduce_mean(policy_distribution.entropy())
                    entropy_loss = self.ent_coef * -entropy
                    self.policy_network.add_loss(entropy_loss)
            self.add_metric("policy_loss", policy_loss, query=query)

            # minimize total policy loss
            total_policy_loss = self.policy_network.build_total_loss()
            optimize = self.policy.optimize_loss(total_policy_loss, name=query)

            # summaries
            self.summary.add_scalar("advantage", tf.reduce_mean(advantage), query=query)
            if self.ent_coef:
                self.summary.add_scalar("entropy", entropy, query=query)
                self.summary.add_scalar("entropy_loss", entropy_loss, query=query)
            self.summary.add_histogram("confidence", policy_distribution.prob(action), query=query)

            if self.output.discrete:
                probs = policy_distribution.probs
                probs = tf.transpose(probs)
                for i in range(probs.shape[0]):
                    self.summary.add_histogram(f"prob_{i}", probs[i], query=query)

        return optimize

    def build_policy_loss(self, action, advantage, query=None):
        # build -log(P(y))
        policy_distribution = self.policy.network.get_distribution_layer()
        neg_log_prob = policy_distribution.neg_log_prob(action)

        # summary
        self.summary.add_scalar("neg_log_prob", tf.reduce_mean(neg_log_prob), query=query)

        # build policy loss
        return tf.reduce_mean(neg_log_prob * advantage)

    def get_baseline_scope(self):
        return "baseline"

    def build_baseline(self, state):
        # build optional baseline
        has_baseline = self.V_definition or self.simple_baseline
        if has_baseline:
            query = "predict"
            with tf.variable_scope(self.get_baseline_scope()):
                if self.V_definition:
                    # build V-network baseline
                    V_network = self.build_V(state, query=query)
                    baseline = V_network.outputs

                    # optimize V-network
                    V_target = self.create_feed("V_target", shape=(None,), query=query)
                    V_loss = self.optimize_V(V_target)

                    if self.V_coef is not None:
                        V_loss *= self.V_coef

                    # add V-loss to policy
                    self.policy_network.add_loss(V_loss)

                    # summary
                    self.add_metric("V", tf.reduce_mean(baseline), query=query)
                elif self.simple_baseline:
                    # Simple baseline  (FIXME - One of Woj's suggestions.  ask him.)
                    baseline = tf.reduce_sum(state)  # TODO - some function of non-params

                    # summary
                    self.add_metric("V", tf.reduce_mean(baseline), query=query)

    def calculate_targets(self, episode):
        # gather rollout data
        rewards = episode["reward"]
        dones = episode["done"]
        notdone = (1 - dones[-1])
        trajectory_length = len(rewards)
        future_reward = 0

        # get available V
        has_V = "V" in episode
        if has_V:
            V = episode["V"]

            # bootstrap incomplete episodes with estimated value
            if self.V_bootstrap and notdone:
                # TODO - implement last_V look-ahead paradigm used by openai/baselines
                last_V = self.fetch_V(episode["next_state"][-1])
                future_reward = last_V

        # calculate discount rewards & advantage for each step
        targets = np.zeros(trajectory_length, dtype=np.float32)
        last_target = 0
        for i in reversed(range(trajectory_length)):
            if self.gae_lambda is not None:
                # Generalized Advantage Estimate w/ Lambda
                delta = rewards[i] + self.gamma * future_reward * notdone - V[i]
                last_target = delta + self.gamma * self.gae_lambda * notdone * last_target

                notdone = (1 - dones[i])
                future_reward = V[i]
            else:
                # Monte Carlo discount rewards
                # FIXME - I don't think I *need* the (1 - dones[i]) term here...
                last_target = rewards[i] + self.gamma * future_reward * (1 - dones[i])
                future_reward = last_target

            targets[i] = last_target

        # calculate V-target (TD-target) and advantage (TD-error)
        if has_V:
            if self.gae_lambda is None:
                V_target = targets
                advantage = targets - V
            else:
                advantage = targets
                V_target = advantage + V

        # normalize advantage
        mean = np.mean(advantage)
        advantage = advantage - mean
        std = np.std(advantage)
        if std > 0:
            advantage /= std

        # calculate advantage
        episode["advantage"] = advantage
        episode["V_target"] = V_target

    def process_episode(self, episode):
        if not super().process_episode(episode):
            return False

        # compute discounted rewards / advantages
        self.calculate_targets(episode)

        return True

    def prepare_feeds(self, query, feed_map):
        super().prepare_feeds(query, feed_map)

        if self.is_optimize(query):
            # feed discounted rewards
            feed_map["advantage"] = self.batch["advantage"]
            feed_map["V_target"] = self.batch["V_target"]

    def get_optimize_query(self, batch):
        query = super().get_optimize_query(batch)

        # optimize baseline
        if self.V_definition:
            query.append("V_optimize")

        return query
