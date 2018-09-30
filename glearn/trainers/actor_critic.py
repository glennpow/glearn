import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.networks import load_network


class ActorCriticTrainer(Trainer):
    def __init__(self, config, policy, critic, gamma=0.95, **kwargs):
        # get basic params
        self.critic_definition = critic
        self.gamma = gamma

        super().__init__(config, policy, **kwargs)

        # actor critic only works for RL
        assert(self.reinforcement)

    def init_optimizer(self):
        # build critic network
        policy = self.policy
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
            policy.set_fetch("advantage", advantage)

            # critic loss minimizes advantage
            critic_loss = tf.reduce_mean(tf.square(advantage))
            self.summary.add_scalar("critic_loss", critic_loss, "evaluate")
            # policy.set_fetch("critic_loss", critic_loss, "evaluate")

            # optimize loss
            critic_optimize = self.optimize_loss(critic_loss, self.critic_definition)
            policy.set_fetch("critic_optimize", critic_optimize)

        # build duplicate "old" actor network
        with tf.name_scope('old_actor'):
            # duplicate policy network
            assert hasattr(policy, "network")
            actor_network = policy.network
            actor_definition = actor_network.definition
            self.old_actor_network = load_network("old_actor", policy, actor_definition,
                                                  trainable=False)
            actor_inputs = policy.get_feed("X")
            self.old_actor_network.build(actor_inputs)

            # build old actor update
            actor_vars = actor_network.get_variables()
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
            # old_actor_prob = self.old_actor_network.prob(past_action) + 1e-5
            # ratio = actor_network.prob(past_action) / old_actor_prob
            actor_log_prob = actor_network.log_prob(past_action)
            old_actor_log_prob = self.old_actor_network.log_prob(past_action)
            ratio = tf.exp(actor_log_prob - old_actor_log_prob)
            surr = ratio * past_advantage

            # clipped surrogate objective
            EPSILON = 0.2
            clipped_surr = tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * past_advantage
            actor_loss = -tf.reduce_mean(tf.minimum(surr, clipped_surr))
            self.summary.add_scalar("loss", actor_loss, "evaluate")
            # policy.set_fetch("actor_loss", actor_loss, "evaluate")

            # optimize the surrogate loss
            optimize = self.optimize_loss(actor_loss)
            policy.set_fetch("optimize", optimize)

    def prepare_feeds(self, graphs, feed_map):
        if "advantage" in graphs or "critic_optimize" in graphs:
            # build critic feed map with rewards
            reward = [e.discounted_reward for e in self.batch.info["transitions"]]
            feed_map["discounted_reward"] = np.array(reward)[:, np.newaxis]

        elif "optimize" in graphs:
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
