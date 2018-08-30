import numpy as np
import tensorflow as tf
from policies.policy import Policy
from policies.layers import add_fc


class PolicyGradient(Policy):
    def __init__(self, learning_rate=2e-4, discount_factor=0.95, hidden_depth=2, hidden_size=10,
                 **kwargs):
        self.learning_rate = learning_rate  # lamdba λ
        self.discount_factor = discount_factor  # gamma γ
        self.hidden_depth = hidden_depth
        self.hidden_size = hidden_size

        super().__init__(**kwargs)

    def init_model(self):
        # create feed placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

            discounted_rewards = tf.placeholder(tf.float32, (None, ), name="gamma")
            self.set_feed("gamma", discounted_rewards, ["optimize", "evaluate"])

        # prepare inputs
        input_size = self.input.size  # FIXME - can we infer this from inputs?
        inputs = tf.reshape(inputs, (-1, input_size))
        layer = inputs

        # create hidden layers
        for i in range(self.hidden_depth):
            layer, info = add_fc(self, layer, input_size, self.hidden_size)
            input_size = layer.shape[1]

        # create output layer
        layer, info = add_fc(self, layer, input_size, self.output.size, activation=tf.nn.softmax)

        # store prediction
        self.set_fetch("predict", layer, ["predict", "evaluate"])

        # calculate loss
        with tf.name_scope('loss'):
            logits = info["Z"]
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
            loss = tf.reduce_mean(neg_log_p * discounted_rewards)
            self.set_fetch("loss", loss, "evaluate")

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.set_fetch("optimize", optimizer.minimize(loss), graph="optimize")

    def prepare_feed_map(self, graph, data, feed_map):
        if graph == "optimize" or graph == "evaluate":
            # normalized discount rewards
            # FIXME - I think this will be broken!  (rewards is no longer part of data)
            discounted_rewards = np.zeros_like(data.rewards)
            accum = 0
            for t in reversed(range(len(data.rewards))):
                accum = accum * self.discount_factor + data.rewards[t]
                discounted_rewards[t] = accum
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            feed_map["gamma"] = discounted_rewards

        return feed_map
