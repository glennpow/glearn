import tensorflow as tf
from glearn.trainers.advantage_actor_critic import AdvantageActorCriticTrainer


# FIXME - old broken code


class PPOTrainer(AdvantageActorCriticTrainer):
    def __init__(self, config, critic, clip_epsilon=0.2, **kwargs):
        self.clip_epsilon = clip_epsilon

        super().__init__(config, critic, **kwargs)

    def on_policy(self):
        return True  # FIXME - DDPG vs. SAC vs. PPO etc.

    def init_actor(self):
        policy = self.policy
        self.policy_network = policy.network

        # build duplicate target policy network
        self.target_policy_network = self.policy_network.clone("target_policy")

        with tf.name_scope("target_policy_optimize"):
            # build target policy update
            policy_vars = self.policy_network.global_variables()
            target_policy_vars = self.target_policy_network.global_variables()
            policy_updates = [tp.assign(p) for p, tp in zip(policy_vars, target_policy_vars)]
            policy_update = tf.group(*policy_updates, name="policy_update")
            self.add_fetch("policy_update", policy_update)

        # build policy optimization
        query = "policy_optimize"
        with tf.name_scope(query):
            with tf.name_scope("loss"):
                action = self.get_feed("Y")
                advantage = self.get_fetch("advantage")

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
                self.add_fetch("policy_loss", policy_loss, "evaluate")
            self.summary.add_scalar("policy_loss", policy_loss)

            # optimize the policy loss
            optimize = self.optimize_loss(policy_loss, query, update_global_step=False)

            self.add_fetch(query, optimize)
