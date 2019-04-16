import tensorflow as tf
from .reinforcement import ReinforcementTrainer
from glearn.networks import load_network


# TODO - broken, FIXME


class SoftActorCriticTrainer(ReinforcementTrainer):
    def __init__(self, config, Q, V, Q_count=2, tau=1e-2, ent_coef=1e-6, gamma=0.95,
                 **kwargs):
        # get basic params
        self.Q_definition = Q
        self.V_definition = V
        self.Q_count = Q_count
        self.tau = tau
        self.ent_coef = ent_coef
        self.gamma = gamma

        super().__init__(config, **kwargs)

        # self.policy_scope = "actor"

    def on_policy(self):
        return False

    def build_trainer(self):
        self.build_critic()
        self.build_actor()

    def build_Q(self):
        # build Q-networks
        state = self.get_feed("X")
        action = self.get_feed("Y")
        Q_inputs = tf.concat([state, action], 1)
        self.Q_networks = []
        for i in range(self.Q_count):
            Q_index = f"Q_{i + 1}"
            Q_network = load_network(Q_index, self, self.Q_definition)
            Q_network.build_predict(Q_inputs)
            self.Q_networks.append(Q_network)

            self.add_fetch(Q_index, Q_network.outputs)

    def build_V(self):
        # build V-network
        state = self.get_feed("X")
        self.V_network = load_network(f"V", self, self.V_definition)
        V = self.V_network.build_predict(state)
        self.add_fetch("V", V)

        # build target V-network
        self.target_V_network = self.clone_network(self.V_network, "target_V")
        self.add_fetch("target_V", self.target_V_network.outputs)

    def build_critic(self):
        with tf.variable_scope("critic"):
            # build Q- and V-networks
            self.build_Q()
            self.build_V()

            # build Q-target
            with tf.variable_scope("Q_target"):
                reward = self.get_or_create_feed("reward", shape=(None,))
                done = self.get_or_create_feed("done", shape=(None,))
                target_V = self.get_or_create_feed("target_V", shape=(None,))
                gamma = tf.constant(self.gamma, name="gamma")
                Q_target = reward + (1 - done) * gamma * target_V

            # build Q-loss and optimize
            Q_optimizes = []
            for i in range(self.Q_count):
                query = f"Q_{i + 1}_optimize"
                with tf.variable_scope(query):
                    # build Q-loss
                    with tf.variable_scope("loss"):
                        Q_i_network = self.Q_networks[i]
                        Q_i = Q_i_network.outputs
                        Q_loss = tf.reduce_mean(tf.squared_difference(Q_i, Q_target))
                    self.summary.add_scalar(f"Q_{i + 1}_loss", Q_loss)

                    # optimize the Q-loss
                    optimize = Q_i_network.optimize_loss(Q_loss)
                    Q_optimizes.append(optimize)

                # Q summaries
                self.summary.add_scalar(f"Q_{i + 1}", tf.reduce_mean(Q_i))

            self.add_fetch("Q_update", tf.group(Q_optimizes, name="Q_update"))

            # build V-loss and optimize
            query = "V_optimize"
            with tf.variable_scope(query):
                with tf.variable_scope("loss"):
                    action = self.get_feed("Y")
                    policy_distribution = self.policy.network.get_distribution_layer()

                    # TODO...
                    # with tf.variable_scope('training_alpha'):
                    #     if self.auto_adjusted_alpha:
                    #         loss_alpha = - tf.reduce_mean(self.log_alpha * tf.stop_gradient(
                    #             self.mu_logp + self._entropy_threshold))
                    #         train_alpha_op, grads_alpha = self._build_optimization_op(loss_alpha, [self.log_alpha], 0.001)

                    #         self.losses += [loss_alpha]
                    #         self.train_ops += [train_alpha_op]

                    # build entropy
                    entropy = tf.reduce_mean(policy_distribution.neg_log_prob(action))
                    self.entropy_factor = self.ent_coef * entropy
                    self.summary.add_scalar("entropy", entropy)

                    # build Q-min
                    Q_min = tf.reduce_min([Q_network.outputs for Q_network in self.Q_networks])

                    # build V-loss
                    V_target = tf.stop_gradient(Q_min + self.entropy_factor)
                    V = self.V_network.outputs
                    V_loss = tf.reduce_mean(tf.squared_difference(V, V_target))
                self.summary.add_scalar("V_loss", V_loss)

                # optimize the V-loss
                V_optimize = self.V_network.optimize_loss(V_loss)

            # V summaries
            self.summary.add_scalar("V", tf.reduce_mean(self.V_network.outputs))
            self.summary.add_scalar("target_V", tf.reduce_mean(self.target_V_network.outputs))

            with tf.control_dependencies([V_optimize]):
                # build target network update
                target_V_update = self.target_V_network.update(self.V_network, self.tau)

            self.add_fetch("V_update", tf.group([V_optimize, target_V_update], name="V_update"))

    def build_actor(self):
        with tf.variable_scope("actor"):
            # build policy optimization
            query = "policy_optimize"
            with tf.variable_scope(query):
                with tf.variable_scope("loss"):
                    # # policy loss
                    Q_1 = tf.stop_gradient(self.Q_networks[0].outputs)
                    policy_loss = -tf.reduce_mean(Q_1 + self.entropy_factor)
                    self.add_fetch("policy_loss", policy_loss, "evaluate")
                self.summary.add_scalar("policy_loss", policy_loss)

                # optimize the policy loss
                policy_optimize = self.policy.network.optimize_loss(policy_loss)

            self.add_fetch("policy_update", policy_optimize)

    def fetch_Q(self, index=None, states=None, actions=None, squeeze=False):
        query = "Q" if index is None else f"Q_{index}"

        # default state
        if states is None:
            states = self.get_feed("X")

        # default action
        if actions is None:
            actions = self.get_feed("Y")

        # fetch
        feed_map = {"X": states, "Y": actions}
        return self.fetch(query, feed_map, squeeze=squeeze)

    def fetch_Q_min(self, states=None, actions=None, squeeze=False):
        # fetch Q_min
        return self.fetch_Q("min", states=states, actions=actions, squeeze=squeeze)

    def fetch_V(self, query=None, states=None, squeeze=False):
        # default query
        if query is None:
            query = "V"

        # default state
        if states is None:
            states = self.get_feed("X")

        # fetch
        feed_map = {"X": states}
        return self.fetch(query, feed_map, squeeze=squeeze)

    def optimize(self, batch):
        feed_map = batch.get_feeds()

        # get next actions
        sampled_actions = self.fetch("predict", {"X": feed_map["X"]})

        # get V-target
        target_V = self.fetch_V("target_V", batch["next_state"], squeeze=True)

        # optimize critic Q-networks
        Q_feed_map = {
            "X": feed_map["X"],
            "Y": feed_map["Y"],
            "target_V": target_V,
            "reward": batch["reward"],
            "done": batch["done"],
        }
        Q_results = self.run("Q_update", Q_feed_map)

        # optimize critic V-networks
        V_feed_map = {
            "X": feed_map["X"],
            "Y": sampled_actions,
        }
        V_results = self.run("V_update", V_feed_map)

        # optimize actor policy
        policy_feed_map = {
            "X": feed_map["X"],
            "Y": sampled_actions,
        }
        policy_results = self.run("policy_update", policy_feed_map)

        return {**Q_results, **V_results, **policy_results}

    def prepare_feeds(self, query, feed_map):
        super().prepare_feeds(query, feed_map)

        if "evaluate" in query:
            # get V-target
            target_V = self.fetch_V("target_V", self.batch["next_state"], squeeze=True)

            # add required feeds
            feed_map["target_V"] = target_V
            feed_map["reward"] = self.batch["reward"]
            feed_map["done"] = self.batch["done"]
