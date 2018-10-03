import tensorflow as tf
from glearn.networks import load_network
from glearn.trainer.actor_network import ActorCriticTrainer


class PPOTrainer(ActorCriticTrainer):
    def init_actor(self):
        policy = self.policy
        assert hasattr(policy, "network")
        self.actor_network = policy.network

        # build duplicate "old" actor network
        with tf.name_scope('old_actor'):
            # duplicate policy network
            actor_definition = self.actor_network.definition
            self.old_actor_network = load_network("old_actor", policy, actor_definition,
                                                  trainable=False)
            actor_inputs = policy.get_feed("X")
            self.old_actor_network.build(actor_inputs)

            # build old actor update
            actor_vars = self.actor_network.get_variables()
            old_actor_vars = self.old_actor_network.get_variables()
            actor_update = [oldp.assign(p) for p, oldp in zip(actor_vars, old_actor_vars)]
            policy.set_fetch("actor_update", actor_update)

        # build actor optimization
        with tf.name_scope('optimize'):
            action_shape = (None,) + self.output.shape
            past_action = policy.create_feed("past_action", "optimize", action_shape)
            past_advantage = policy.create_feed("past_advantage", "optimize", (None, 1))
            # past_advantage = advantage

            # surrogate loss
            actor_distribution = self.actor_network.get_distribution()
            old_actor_distribution = self.old_actor_network.get_distribution()
            actor_log_prob = actor_distribution.log_prob(past_action)
            old_actor_log_prob = old_actor_distribution.log_prob(past_action)
            ratio = tf.exp(actor_log_prob - old_actor_log_prob)
            surr = ratio * past_advantage

            # clipped surrogate objective
            EPSILON = 0.2
            clipped_surr = tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * past_advantage
            actor_loss = -tf.reduce_mean(tf.minimum(surr, clipped_surr))

            # entropy
            actor_loss -= self.ent_coef * actor_distribution.entropy()

            policy.set_fetch("actor_loss", actor_loss, "evaluate")

            # optimize the surrogate loss
            optimize = self.optimize_loss(actor_loss)
            policy.set_fetch("optimize", optimize)

        # summaries
        self.summary.add_scalar("loss", actor_loss, "evaluate")
