import tensorflow as tf
from glearn.trainers.td import TDTrainer
from glearn.networks import load_network
from glearn.utils.collections import intersects


class ActorCriticTrainer(TDTrainer):
    def __init__(self, config, policy, critic, ent_coef=1e-5, **kwargs):
        # get basic params
        self.critic_definition = critic
        self.ent_coef = ent_coef

        super().__init__(config, policy, **kwargs)

        # actor critic only works for RL
        assert(self.reinforcement)

    def init_optimizer(self):
        self.init_critic()

        self.init_actor()

        super().init_optimizer()

    def init_critic(self):
        policy = self.policy

        # build critic network
        with tf.name_scope("value"):
            self.critic_network = load_network("critic", policy, self.critic_definition)
            critic_inputs = policy.get_feed("X")
            critic_value = self.critic_network.build_predict(critic_inputs)
            policy.set_fetch("value", critic_value)

        value_loss = self.build_value_loss(critic_value)
        self.summary.add_scalar("value", tf.reduce_mean(critic_value), "evaluate")

        # build advantage and critic optimization
        graph = "value_optimize"
        with tf.name_scope(graph):
            optimize = self.optimize_loss(value_loss, graph, self.critic_definition)

            self.policy.set_fetch(graph, optimize)  # FIXME? does it matter that this is in actor?

    def init_actor(self):
        policy = self.policy
        assert hasattr(policy, "network")
        self.policy_network = policy.network

        # build policy optimization
        entropy_loss = None
        graph = "policy_optimize"
        with tf.name_scope(graph):
            action_shape = (None,) + self.output.shape
            graphs = [graph, "evaluate"]
            past_action = policy.create_feed("past_action", graphs, action_shape)
            past_advantage = policy.create_feed("past_advantage", graphs, (None, 1))

            # policy loss
            policy_distribution = policy.network.get_distribution_layer()
            neg_logp = -policy_distribution.log_prob(past_action)
            policy_loss = neg_logp * past_advantage

            # entropy exploration factor (TODO - could add to losses collection)
            if self.ent_coef > 0:
                entropy = policy_distribution.entropy()
                policy.set_fetch("entropy", entropy, "debug")
                entropy_loss = -self.ent_coef * entropy
                policy_loss += entropy_loss

            # L2 distribution loss (TODO - could add to losses collection)
            l2_loss = None
            if "l2_loss" in policy_distribution.references:
                l2_loss = policy_distribution.references["l2_loss"]
                policy_loss += l2_loss

            policy_loss = tf.reduce_mean(policy_loss)
            policy.set_fetch("policy_loss", policy_loss, "debug")

            # optimize the policy loss
            optimize = self.optimize_loss(policy_loss, graph, update_global_step=False)

            self.policy.set_fetch(graph, optimize)

        # add summaries
        if entropy_loss is not None:
            self.summary.add_scalar("entropy_loss", tf.reduce_mean(entropy_loss), "evaluate")
            self.summary.add_scalar("entropy", tf.reduce_mean(entropy), "evaluate")
        # if hasattr(policy_distribution, "stddev"):
        #     action_stddev = tf.reduce_mean(tf.squeeze(policy_distribution.stddev()))
        #     self.summary.add_scalar("action_stddev", action_stddev, "evaluate")
        self.summary.add_scalar("policy_loss", policy_loss, "evaluate")
        # mu = policy_distribution.references["mu"]
        # sigma = policy_distribution.references["sigma"]
        # self.summary.add_scalar("mu", tf.reduce_mean(mu), "evaluate")
        # self.summary.add_scalar("sigma", tf.reduce_mean(sigma), "evaluate")
        # if l2_loss is not None:
        #     self.summary.add_scalar("l2_loss", l2_loss, "evaluate")

    def prepare_feeds(self, graphs, feed_map):
        if intersects(["policy_optimize", "evaluate"], graphs):
            feed_map["past_action"] = self.batch.outputs
            feed_map["past_advantage"] = self.past_advantage

        return super().prepare_feeds(graphs, feed_map)

    def run(self, graphs, feed_map={}, **kwargs):
        if not isinstance(graphs, list):
            graphs = [graphs]

        if intersects(["policy_optimize", "evaluate"], graphs):
            # get advantage
            self.past_advantage = super().fetch("advantage", feed_map)

        if "policy_optimize" in graphs:
            # optimize critic value network as well
            graphs += ["value_optimize"]

        results = super().run(graphs, feed_map, **kwargs)

        return results
