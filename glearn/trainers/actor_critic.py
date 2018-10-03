import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.networks import load_network
from glearn.utils.collections import intersects


class ActorCriticTrainer(Trainer):
    def __init__(self, config, policy, critic, ent_coef=1e-5, gamma=0.95, **kwargs):
        # get basic params
        self.critic_definition = critic
        self.normalize_advantage = False  # TODO
        self.ent_coef = ent_coef
        self.gamma = gamma

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
            policy.set_fetch("critic_value", critic_value)

        # build advantage and critic optimization
        with tf.name_scope('critic_optimize'):
            # reward = policy.create_feed("reward", ["critic_optimize", "optimize"], (None,))
            # gamma = tf.constant(self.gamma, name="gamma")

            # calculate advantage, using discounted rewards
            # discounted_reward = reward + gamma * critic_value  # this happens out of graph now
            discounted_reward = policy.create_feed("discounted_reward", ["advantage"], (None, 1))
            advantage = discounted_reward - critic_value
            if self.normalize_advantage:  # aborghi implementation
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy.set_fetch("advantage", advantage)

            # critic loss minimizes advantage
            critic_loss = tf.reduce_mean(tf.square(advantage))
            policy.set_fetch("critic_loss", critic_loss, "evaluate")

            # optimize loss
            critic_optimize = self.optimize_loss(critic_loss, self.critic_definition)
            policy.set_fetch("critic_optimize", critic_optimize)

        # summaries
        self.summary.add_scalar("critic_loss", critic_loss, "evaluate")

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
            actor_loss = actor_neg_logp * past_advantage - entropy_factor
            actor_loss = tf.reduce_mean(actor_loss)
            policy.set_fetch("actor_loss", actor_loss, "evaluate")

            # optimize the actor loss
            optimize = self.optimize_loss(actor_loss)
            policy.set_fetch("optimize", optimize)

        # add summaries
        self.summary.add_scalar("loss", actor_loss, "evaluate")

    def prepare_feeds(self, graphs, feed_map):
        if intersects(["advantage", "critic_optimize"], graphs):
            # build critic feed map with rewards
            if self.batch is not None:
                reward = [e.discounted_reward for e in self.batch.info["transitions"]]
                feed_map["discounted_reward"] = np.array(reward)[:, np.newaxis]
            else:
                shape = np.shape(feed_map["X"])[:-1] + (1,)
                feed_map["discounted_reward"] = np.zeros(shape)

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
            # update old actor policy
            super().run("actor_update", feed_map)

            # get advantage
            self.past_advantage = super().fetch("advantage", feed_map)

        results = super().run(graphs, feed_map)

        if is_optimize:
            # optimize critic
            super().run("critic_optimize", feed_map)

        return results

    def process_transition(self, transition):
        # get critic value and apply gamma here, during rollouts
        # TODO - could get this along WITH 'predict' fetch before?
        feed_map = {"X": [transition.observation]}

        critic_value = self.fetch("critic_value", feed_map, squeeze=True)
        transition.discounted_reward = transition.reward + self.gamma * critic_value
