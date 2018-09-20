import numpy as np
import tensorflow as tf
from glearn.policies.policy import Policy


class NNPolicy(Policy):
    def __init__(self, config):
        self.fc_layers = config.get("fc_layers", [128, 128])

        super().__init__(config)

    def init_model(self):
        # create input placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

            dropout = tf.placeholder(tf.float32, (), name="dropout")
            self.set_feed("dropout", dropout)

            # gamma = tf.placeholder(tf.float32, (None, ), name="gamma")
            # self.set_feed("gamma", gamma, ["optimize", "evaluate"])

        # prepare inputs
        input_size = self.input.size  # FIXME - can we infer this from inputs?
        inputs = tf.reshape(inputs, (-1, input_size))
        layer = tf.cast(inputs, tf.float32)

        # create fully connected layers
        input_size = np.prod(layer.shape[1:])
        for i, fc_size in enumerate(self.fc_layers):
            layer, info = self.add_fc(layer, input_size, fc_size, reshape=i == 0,
                                      keep_prob=dropout)
            input_size = layer.shape[1]

        # create output layer
        layer, info = self.add_fc(layer, input_size, self.output.size,
                                  activation=tf.nn.softmax)

        # store prediction
        self.set_fetch("predict", layer, ["predict", "evaluate"])

        # calculate loss
        with tf.name_scope('loss'):
            logits = info["Z"]
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
            loss = tf.reduce_mean(neg_log_p)  # * discounted_rewards)
            self.set_fetch("loss", loss, "evaluate")

    def prepare_default_feeds(self, graph, feed_map):
        feed_map["dropout"] = 1
        return feed_map
