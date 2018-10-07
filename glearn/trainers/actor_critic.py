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

    def init_critic(self):
        policy = self.policy

        # build critic network
        with tf.name_scope('critic'):
            self.critic_network = load_network("critic", policy, self.critic_definition)
            critic_inputs = policy.get_feed("X")
            critic_value = self.critic_network.build(critic_inputs)
            policy.set_fetch("value", critic_value)

        value_loss = self.init_value_loss(critic_value)

        # build advantage and critic optimization
        with tf.name_scope('value_optimize'):
            # optimize loss
            value_optimize = self.optimize_loss(value_loss, self.critic_definition)
            policy.set_fetch("value_optimize", value_optimize)

    def init_actor(self):
        policy = self.policy
        self.actor_network = policy.network

        # build actor optimization
        with tf.name_scope('optimize'):
            action_shape = (None,) + self.output.shape
            past_action = policy.create_feed("past_action", "optimize", action_shape)
            past_advantage = policy.create_feed("past_advantage", "optimize", (None, 1))
            # past_advantage = advantage

            # actor loss
            actor_distribution = policy.network.get_distribution()
            actor_neg_logp = -actor_distribution.log_prob(past_action)
            entropy_factor = self.ent_coef * actor_distribution.entropy()
            policy_loss = actor_neg_logp * past_advantage - entropy_factor
            policy_loss = tf.reduce_mean(policy_loss)
            policy.set_fetch("policy_loss", policy_loss, "evaluate")

            # optimize the actor loss
            policy_optimize = self.optimize_loss(policy_loss)
            policy.set_fetch("optimize", policy_optimize)

        # add summaries
        self.summary.add_scalar("loss", policy_loss, "evaluate")

    def prepare_feeds(self, graphs, feed_map):
        # print(f"intersects-optimize ({graphs})...")
        if intersects("optimize", graphs):
            feed_map["past_action"] = self.batch.outputs
            feed_map["past_advantage"] = self.past_advantage

        return super().prepare_feeds(graphs, feed_map)

    def run(self, graphs, feed_map={}):
        if not isinstance(graphs, list):
            graphs = [graphs]

        is_optimize = "optimize" in graphs
        if is_optimize:
            # get advantage
            self.past_advantage = super().fetch("advantage", feed_map)

        results = super().run(graphs, feed_map)

        if is_optimize:
            # optimize critic
            super().run("value_optimize", feed_map)

        return results
