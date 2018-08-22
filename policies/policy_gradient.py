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
        # create placeholders
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, (None,) + self.input.shape, name="X")
            self.outputs = tf.placeholder(tf.float32, (None,) + self.output.shape, name="Y")
            self.discounted_rewards = tf.placeholder(tf.float32, (None, ), name="V")

        # prepare inputs
        input_size = self.input.size  # FIXME - can we infer this from inputs?
        inputs = tf.reshape(self.inputs, (-1, input_size))
        layer = (inputs, None)

        # create hidden layers
        for i in range(self.hidden_depth):
            layer = add_fc(self, layer[0], input_size, self.hidden_size)
            input_size = layer[0].shape[1]

        # create output layer
        layer = add_fc(self, layer[0], input_size, self.output.size,
                       activation=tf.nn.softmax)

        # softmax
        logits = layer[1]["Z"]
        labels = self.outputs
        act = layer[0]
        self.act_graph["act"] = act

        # calculate loss
        with tf.name_scope('loss'):
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_p * self.discounted_rewards)
            self.evaluate_graph["loss"] = loss

        # minimize loss
        with tf.name_scope('optimize'):
            optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            self.optimize_graph["optimize"] = optimize

    def optimize_feed(self, batch, feed_dict):
        # normalized discount rewards
        discounted_rewards = np.zeros_like(batch.rewards)
        accum = 0
        for t in reversed(range(len(batch.rewards))):
            accum = accum * self.discount_factor + batch.rewards[t]
            discounted_rewards[t] = accum
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        feed_dict[self.discounted_rewards] = discounted_rewards

        return feed_dict
