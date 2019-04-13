import tensorflow as tf
from .reinforcement import ReinforcementTrainer


class DeepQNetworkTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, Q=None, Q_count=1, target_update_steps=10, tau=None,
                 **kwargs):
        self.gamma = gamma
        self.Q_definition = Q
        self.Q_count = Q_count
        self.target_update_steps = target_update_steps
        self.tau = tau

        super().__init__(config, **kwargs)

    def on_policy(self):
        return False

    def build_trainer(self):
        # get the inputs
        state = self.get_feed("X")
        action = self.get_feed("Y")
        next_state = self.get_or_create_feed("next_state", (None,) + self.input.shape)
        reward = self.get_or_create_feed("reward", shape=(None,))
        done = self.get_or_create_feed("done", (None,))

        # build Q-network, action prediction and actual
        # with tf.variable_scope("DQN"):
        Q_network = self.build_Q(state)
        Q = Q_network.outputs

        with self.variable_scope(Q_network.scope):
            predict = tf.expand_dims(tf.argmax(Q, axis=-1), axis=-1)
            self.add_fetch("predict", predict)

        # build Target-Q-Network
        target_Q_network = Q_network.clone("target_Q", inputs=next_state)
        target_Q = target_Q_network.outputs

        with self.variable_scope("Q_optimize"):
            # the prediction by the primary Q network for the actual actions.
            with self.variable_scope("Q_acted"):
                action_one_hot = tf.one_hot(action, self.output.size, 1.0, 0.0, name='action_ones')
                Q_acted = tf.reduce_sum(Q * action_one_hot, reduction_indices=-1, name='Q_acted')

            # the optimization target defined by the Bellman equation and the target network.
            with self.variable_scope("Q_target"):
                target_Q_predict = tf.reduce_max(target_Q, axis=-1)
                Q_target = reward + (1 - done) * self.gamma * target_Q_predict

        # minimize mean square error
        mode = "mse"  # TODO "huber"
        weight = None  # TODO - priority weighting
        Q_optimize, Q_loss = Q_network.optimize_error(Q_target, Q_acted, mode=mode, weights=weight)

        # build target network update
        with tf.control_dependencies([Q_optimize]):
            target_Q_network.update(Q_network, tau=self.tau)

        # summaries
        self.summary.add_histogram("predict", predict, query="predict")
        with self.variable_scope("Q_optimize"):
            self.add_metric("target_Q", target_Q, query="Q_optimize")
            self.add_metric("Q", Q_acted, query="Q_optimize")

        return Q_optimize

    def prepare_feeds(self, query, feed_map):
        super().prepare_feeds(query, feed_map)

        if self.is_optimize(query) or self.is_evaluate(query):
            feed_map["next_state"] = self.batch["next_state"]
            feed_map["reward"] = self.batch["reward"]
            feed_map["done"] = self.batch["done"]

    def get_optimize_query(self, batch):
        query = ["Q_optimize"]

        # update target network at desired step interval
        if self.target_update_steps is None or \
           self.current_global_step % self.target_update_steps == 0:
            query.append("target_Q_update")

        return query
