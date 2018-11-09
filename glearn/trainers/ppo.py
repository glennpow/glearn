import tensorflow as tf
from glearn.networks import load_network
from glearn.trainer.actor_network import ActorCriticTrainer


class PPOTrainer(ActorCriticTrainer):
    def __init__(self, clip_epsilon=0.2):
        self.clip_epsilon = clip_epsilon

    def init_actor(self):
        policy = self.policy
        # assert hasattr(policy, "network")
        self.policy_network = policy.network

        # build duplicate "old" policy network
        with tf.name_scope('old_policy'):
            # duplicate policy network
            policy_definition = self.policy_network.definition
            self.old_policy_network = load_network("old_policy", policy, policy_definition,
                                                   trainable=False)
            policy_inputs = policy.get_feed("X")
            self.old_policy_network.build(policy_inputs)

            # build old policy update
            policy_vars = self.policy_network.get_variables()
            old_policy_vars = self.old_policy_network.get_variables()
            policy_update = [oldp.assign(p) for p, oldp in zip(policy_vars, old_policy_vars)]
            policy.set_fetch("policy_update", policy_update)

        # build policy optimization
        with tf.name_scope('policy_optimize'):
            action_shape = (None,) + self.output.shape
            past_action = policy.create_feed("past_action", "policy_optimize", action_shape)
            past_advantage = policy.create_feed("past_advantage", "policy_optimize", (None, 1))
            # past_advantage = advantage

            # surrogate loss
            policy_distribution = self.policy_network.get_distribution_layer()
            old_policy_distribution = self.old_policy_network.get_distribution_layer()
            policy_log_prob = policy_distribution.log_prob(past_action)
            old_policy_log_prob = old_policy_distribution.log_prob(past_action)
            ratio = tf.exp(policy_log_prob - old_policy_log_prob)
            surr = ratio * past_advantage

            # clipped surrogate objective
            epsilon = self.clip_epsilon
            clipped_surr = tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * past_advantage
            policy_loss = -tf.reduce_mean(tf.minimum(surr, clipped_surr))

            # entropy
            policy_loss -= self.ent_coef * policy_distribution.entropy()

            policy.set_fetch("policy_loss", policy_loss, "evaluate")

        # optimize the surrogate loss
        self.optimize_loss("policy_optimize", policy_loss)

        # summaries
        self.summary.add_scalar("loss", policy_loss, "evaluate")
