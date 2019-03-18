import tensorflow as tf
from glearn.trainers.temporal_difference import TemporalDifferenceTrainer


# FIXME - old broken code


class ActorCriticTrainer(TemporalDifferenceTrainer):
    def __init__(self, config, critic, **kwargs):
        # get basic params
        self.critic_definition = critic

        super().__init__(config, **kwargs)

        # actor critic only works for RL
        assert(self.has_env)

    def build_trainer(self):
        self.init_critic()

        self.init_actor()

    def init_critic(self):
        policy = self.policy

        # build critic network
        self.critic_network = self.build_network("value", policy, self.critic_definition)
        critic_inputs = self.get_feed("X")
        critic_value = self.critic_network.build_predict(critic_inputs)
        self.add_fetch("value", critic_value)
        with tf.name_scope("value/"):
            avg_critic_value = tf.reduce_mean(critic_value)

        # build advantage and critic optimization
        query = "value_optimize"
        with tf.name_scope(query):
            value_loss = self.build_value_loss(critic_value)

            self.critic_network.optimize_loss(value_loss, name=query)

            self.summary.add_scalar("value", avg_critic_value)

    def init_actor(self):
        policy = self.policy
        self.policy_network = policy.network

        # build policy optimization
        query = "policy_optimize"
        with tf.name_scope(query):
            with tf.name_scope("loss"):
                action = self.get_feed("Y")
                advantage = self.get_fetch("advantage")

                # actor loss
                policy_distribution = policy.network.get_distribution_layer()
                neg_logp = policy_distribution.neg_log_prob(action)
                actor_loss = tf.reduce_mean(neg_logp * advantage)
                self.policy_network.add_loss(actor_loss)

                # total policy loss
                policy_loss = self.policy_network.get_total_loss()
                policy_loss = tf.reduce_mean(policy_loss)

            # summary
            self.add_evaluate_metric("policy_loss", policy_loss)

            # optimize the policy loss
            self.optimize_loss(policy_loss, networks=[self.policy_network], name=query)

    def run(self, queries, feed_map={}, **kwargs):
        if not isinstance(queries, list):
            queries = [queries]

        if "policy_optimize" in queries:
            # optimize critic value network as well
            queries += ["value_optimize"]

        results = super().run(queries, feed_map, **kwargs)

        return results
