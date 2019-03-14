import tensorflow as tf
from glearn.trainers.temporal_difference import TemporalDifferenceTrainer
from glearn.networks import load_network


# FIXME - old broken code


class AdvantageActorCriticTrainer(TemporalDifferenceTrainer):
    def __init__(self, config, critic, normalize_advantage=False, **kwargs):
        # get basic params
        self.critic_definition = critic
        self.normalize_advantage = normalize_advantage

        super().__init__(config, **kwargs)

        # actor critic only works for RL
        assert(self.has_env)

    def build_trainer(self):
        self.init_critic()

        self.init_actor()

    def init_critic(self):
        # build critic network
        self.critic_network = load_network("value", self.context, self.critic_definition)
        critic_inputs = self.get_feed("X")
        critic_value = self.critic_network.build_predict(critic_inputs)
        self.add_fetch("value", critic_value)
        with tf.name_scope("value/"):
            avg_critic_value = tf.reduce_mean(critic_value)

        # build advantage and critic optimization
        query = "value_optimize"
        with tf.name_scope(query):
            value_loss = self.build_value_loss(critic_value)

            optimize = self.optimize_loss(value_loss, query, self.critic_definition)

            self.add_fetch(query, optimize)

            self.summary.add_scalar("value", avg_critic_value)

    def build_value_loss(self, value):
        with tf.name_scope("loss"):
            # calculate advantage estimate (td_error), using discounted rewards (td_target)
            discounted_reward = self.get_feed("td_target")
            advantage = discounted_reward - value

            # aborghi implementation
            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            self.add_fetch("advantage", advantage)

            # value loss minimizes squared advantage
            value_loss = tf.reduce_mean(tf.square(advantage))
            self.add_fetch("value_loss", value_loss, "evaluate")

            # averages
            discounted_reward = tf.reduce_mean(discounted_reward)
            advantage = tf.reduce_mean(advantage)

        # summaries
        self.summary.add_scalar("value_loss", value_loss)
        self.summary.add_scalar("discount_reward", discounted_reward)
        self.summary.add_scalar("advantage", advantage)

        return value_loss

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
                self.add_fetch("policy_loss", policy_loss, "evaluate")
            self.summary.add_scalar("policy_loss", policy_loss)

            # optimize the policy loss
            optimize = self.optimize_loss(policy_loss, query, update_global_step=False)

            self.add_fetch(query, optimize)

    def run(self, queries, feed_map={}, **kwargs):
        if not isinstance(queries, list):
            queries = [queries]

        if "policy_optimize" in queries:
            # optimize critic value network as well
            queries += ["value_optimize"]

        results = super().run(queries, feed_map, **kwargs)

        return results
