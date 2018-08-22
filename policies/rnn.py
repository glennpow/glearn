import tensorflow as tf
from policies.policy import Policy
from policies.layers import add_fc


class RNN(Policy):
    def __init__(self, learning_rate=2e-4, recurrent_depth=2, recurrent_size=128,
                 forget_bias=1, **kwargs):
        self.learning_rate = learning_rate  # lamdba λ
        self.recurrent_depth = recurrent_depth
        self.recurrent_size = recurrent_size
        self.forget_bias = forget_bias

        super().__init__(**kwargs)

    def init_model(self):
        # infer timesteps
        self.timesteps = self.input.shape[1]

        # create placeholders
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, (None,) + self.input.shape, name="X")
            self.outputs = tf.placeholder(tf.float32, (None,) + self.output.shape, name="Y")

        # prepare inputs
        layer = tf.unstack(self.inputs, self.timesteps, 1)

        # define lstm cell and build RNN
        cell = tf.rnn.BasicLSTMCell(self.recurrent_size, forget_bias=self.forget_bias)
        for i in range(self.recurrent_depth):
            layer, states = tf.rnn.static_rnn(cell, layer, dtype=tf.float32)

        # create output layer
        input_size = self.recurrent_size
        layer = add_fc(self, layer, input_size, self.output.size, activation=tf.nn.softmax)

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
            # optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            optimize = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize_graph["optimize"] = optimize.minimize(loss)
