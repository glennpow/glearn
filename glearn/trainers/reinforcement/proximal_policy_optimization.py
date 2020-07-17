import tensorflow as tf
from .advantage_actor_critic import AdvantageActorCriticTrainer
from glearn.utils import tf_utils


class ProximalPolicyOptimizationTrainer(AdvantageActorCriticTrainer):
    def __init__(self, config, clip_epsilon=0.2, **kwargs):
        self.clip_epsilon = clip_epsilon

        super().__init__(config, **kwargs)

    def build_policy_optimize(self, state, action):
        self.policy_network = self.policy.network
        policy_distribution = self.policy_network.get_distribution_layer()
        query = "policy_optimize"

        # build duplicate target policy network and update operation
        self.target_policy_network = self.clone_network(self.policy_network, "target_policy")
        target_policy_distribution = self.target_policy_network.get_distribution_layer()

        # build KL divergence before updating distribution
        kl_divergence = target_policy_distribution.kl_divergence(policy_distribution)

        # perform policy optimization, using loss below
        optimize = super().build_policy_optimize(state, action)

        # update target policy network
        with tf.control_dependencies([optimize, kl_divergence]):
            self.target_policy_network.update(self.policy_network)

        # summaries
        with self.variable_scope(query):
            self.summary.add_scalar("kl_divergence", tf.reduce_mean(kl_divergence), query=query)

        return optimize

    def build_policy_loss(self, action, advantage, query=None):
        policy_distribution = self.policy.network.get_distribution_layer()
        policy_log_prob = policy_distribution.log_prob(action)

        # surrogate loss
        target_policy_distribution = self.target_policy_network.get_distribution_layer()
        target_policy_log_prob = target_policy_distribution.log_prob(action)
        ratio = tf.exp(policy_log_prob - target_policy_log_prob)
        unclipped_surr = ratio * advantage
        if self.debugging:  # HACK
            unclipped_surr = tf_utils.nan_to_num(unclipped_surr)

        # clipped surrogate objective
        epsilon = self.clip_epsilon
        clipped_surr = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage

        # final policy loss
        loss = -tf.reduce_mean(tf.minimum(unclipped_surr, clipped_surr))

        # debug metrics
        prob = policy_distribution.prob(action)
        debug_queries = {
            "action": action,
            "prob": prob,
            "policy_log_prob": policy_log_prob,
            "target_policy_log_prob": target_policy_log_prob,
            "ratio": ratio,
            "unclipped_surr": unclipped_surr,
            "clipped_surr": clipped_surr,
            "loss": loss,
        }
        self.add_debug_metrics(debug_queries)

        return loss

    def get_optimize_query(self, batch):
        query = super().get_optimize_query(batch)

        # update target policy
        query.append("target_policy_update")

        return query
