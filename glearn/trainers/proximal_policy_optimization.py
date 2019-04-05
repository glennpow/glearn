import tensorflow as tf
from glearn.trainers.advantage_actor_critic import AdvantageActorCriticTrainer


class ProximalPolicyOptimizationTrainer(AdvantageActorCriticTrainer):
    def __init__(self, config, clip_epsilon=0.2, **kwargs):
        self.clip_epsilon = clip_epsilon

        super().__init__(config, **kwargs)

    # def build_trainer(self):
    #     super
    #     # build duplicate target policy network
    #     self.target_policy_network = self.policy.network.clone("target_policy")

    #     self.target_policy_network.update(self.policy.network)

    def optimize_policy(self, state, action):
        self.policy_network = self.policy.network
        query = "policy_optimize"

        # build duplicate target policy network
        self.target_policy_network = self.policy_network.clone("target_policy")

        self.target_policy_network.update(self.policy_network)

        # build advantage
        advantage = self.build_expected_return(state, action, query=query)

        with tf.variable_scope(query):
            # build surrogate policy loss
            with tf.name_scope("loss"):
                # build log(P(y))
                policy_distribution = self.policy_network.get_distribution_layer()
                policy_log_prob = policy_distribution.log_prob(action)

                # surrogate loss
                target_policy_distribution = self.target_policy_network.get_distribution_layer()
                # policy_log_prob = policy_distribution.log_prob(action)
                target_policy_log_prob = target_policy_distribution.log_prob(action)
                ratio = tf.exp(policy_log_prob - target_policy_log_prob)
                unclipped_surr = ratio * advantage

                # clipped surrogate objective
                epsilon = self.clip_epsilon
                clipped_surr = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
                policy_loss = -tf.reduce_mean(tf.minimum(unclipped_surr, clipped_surr))
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
            kl_divergence = target_policy_distribution.kl_divergence(policy_distribution)
            self.summary.add_scalar("kl_divergence", tf.reduce_mean(kl_divergence), query=query)
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

    def get_optimize_query(self, batch):
        query = super().get_optimize_query(batch)

        # update target policy
        query.append("target_policy_update")

        return query
