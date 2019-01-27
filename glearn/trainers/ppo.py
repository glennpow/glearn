import tensorflow as tf
from glearn.networks import load_network
from glearn.trainers.actor_critic import ActorCriticTrainer


class PPOTrainer(ActorCriticTrainer):
    def __init__(self, config, policy, critic, clip_epsilon=0.2, **kwargs):
        self.clip_epsilon = clip_epsilon

        super().__init__(config, policy, critic, **kwargs)

    def init_actor(self):
        policy = self.policy
        self.policy_network = policy.network

        # build duplicate target policy network
        policy_definition = self.policy_network.definition
        self.target_policy_network = load_network("target_policy", policy, policy_definition,
                                                  trainable=False)
        policy_inputs = policy.get_feed("X")
        self.target_policy_network.build_predict(policy_inputs)

        with tf.name_scope("target_policy_optimize"):
            # build target policy update
            policy_vars = self.policy_network.global_variables()
            target_policy_vars = self.target_policy_network.global_variables()
            policy_updates = [tp.assign(p) for p, tp in zip(policy_vars, target_policy_vars)]
            policy_update = tf.group(*policy_updates, name="policy_update")
            policy.set_fetch("policy_update", policy_update)

        # build policy optimization
        query = "policy_optimize"
        with tf.name_scope(query):
            with tf.name_scope("loss"):
                action = policy.get_feed("Y")
                advantage = policy.get_fetch("advantage")

                # surrogate loss
                policy_distribution = self.policy_network.get_distribution_layer()
                target_policy_distribution = self.target_policy_network.get_distribution_layer()
                policy_log_prob = policy_distribution.log_prob(action)
                target_policy_log_prob = target_policy_distribution.log_prob(action)
                ratio = tf.exp(policy_log_prob - target_policy_log_prob)
                surr = ratio * advantage

                # clipped surrogate objective
                epsilon = self.clip_epsilon
                clipped_surr = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
                actor_loss = -tf.reduce_mean(tf.minimum(surr, clipped_surr))
                self.policy_network.add_loss(actor_loss)

                # total policy loss
                policy_loss = self.policy_network.get_total_loss()
                policy_loss = tf.reduce_mean(policy_loss)
                policy.set_fetch("policy_loss", policy_loss, "evaluate")
            self.summary.add_scalar("policy_loss", policy_loss, "evaluate")

            # optimize the policy loss
            optimize = self.optimize_loss(policy_loss, query)

            self.policy.set_fetch(query, optimize)
