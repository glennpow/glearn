import numpy as np
import tensorflow as tf
from .layer import NetworkLayer


class DenseLayer(NetworkLayer):
    def __init__(self, network, index, hidden_sizes=[128], activation=tf.nn.relu,
                 weights_initializer=None, biases_initializer=None, weight_decay=None,
                 multiplier=None):
        super().__init__(network, index)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.weight_decay = weight_decay
        self.multiplier = multiplier

    def build(self, inputs):
        # get variables
        self.dropout = self.context.get_or_create_feed("dropout")

        # initializers
        weights_initializer = self.load_initializer(self.weights_initializer)
        self.references["weights_initializer"] = weights_initializer
        biases_initializer = self.load_initializer(self.biases_initializer)
        self.references["weights_initializer"] = biases_initializer

        # create fully connected layers
        input_size = np.prod(inputs.shape[1:])
        y = tf.reshape(tf.cast(inputs, tf.float32), (-1, input_size))
        layers = []
        for hidden_size in self.hidden_sizes:
            # zero means output size
            if hidden_size == 0:
                hidden_size = self.context.output.size

            y = self.dense(y, hidden_size, self.dropout, self.activation,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer,
                           weight_decay=self.weight_decay)
            layers.append(y)
        self.references["layers"] = layers

        # multiplier
        if self.multiplier is not None:
            y *= self.multiplier

        return y

    def build_loss(self, outputs):
        # evaluate continuous loss
        loss = tf.reduce_mean(tf.square(outputs - self.outputs))

        # evaluate accuracy
        accuracy = tf.exp(-loss)

        return loss, accuracy

    def prepare_default_feeds(self, queries, feed_map):
        feed_map["dropout"] = 1
        return feed_map
