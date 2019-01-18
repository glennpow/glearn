import tensorflow as tf
from glearn.trainers.td import TDTrainer
from glearn.networks import load_network
from glearn.utils.collections import intersects


class ActorCriticTrainer(TDTrainer):
    def __init__(self, config, policy, critic, **kwargs):
        # get basic params
        self.critic_definition = critic

        super().__init__(config, policy, **kwargs)

        # actor critic only works for RL
        assert(self.reinforcement)

    def init_optimizer(self):
        self.init_critic()

        self.init_actor()

    def init_critic(self):
        policy = self.policy

        # build critic network
        self.critic_network = load_network("value", policy, self.critic_definition)
        critic_inputs = policy.get_feed("X")
        critic_value = self.critic_network.build_predict(critic_inputs)
        policy.set_fetch("value", critic_value)
        with tf.name_scope("value/"):
            avg_critic_value = tf.reduce_mean(critic_value)

        # build advantage and critic optimization
        family = "value_optimize"
        with tf.name_scope(family):
            value_loss = self.build_value_loss(critic_value)

            optimize = self.optimize_loss(value_loss, family, self.critic_definition)

            self.policy.set_fetch(family, optimize)

            self.summary.add_scalar("value", avg_critic_value, "evaluate")

    def init_actor(self):
        policy = self.policy
        assert hasattr(policy, "network")
        self.policy_network = policy.network

        # build policy optimization
        family = "policy_optimize"
        with tf.name_scope(family):
            with tf.name_scope("loss"):
                action = policy.get_feed("Y")
                past_advantage = policy.get_fetch("advantage")

                # actor loss
                policy_distribution = policy.network.get_distribution_layer()
                neg_logp = policy_distribution.neg_log_prob(action, name="foo")
                actor_loss = tf.reduce_mean(neg_logp * past_advantage)
                self.policy_network.add_loss(actor_loss)

                # total policy loss
                policy_loss = self.policy_network.get_total_loss()
                policy_loss = tf.reduce_mean(policy_loss)
                policy.set_fetch("policy_loss", policy_loss, "evaluate")
            self.summary.add_scalar("policy_loss", policy_loss, "evaluate")

            # optimize the policy loss
            optimize = self.optimize_loss(policy_loss, family, update_global_step=False)

            self.policy.set_fetch(family, optimize)

    def run(self, families, feed_map={}, **kwargs):
        if not isinstance(families, list):
            families = [families]

        if "policy_optimize" in families:
            # optimize critic value network as well
            families += ["value_optimize"]

        results = super().run(families, feed_map, **kwargs)

        return results
