import numpy as np
import tensorflow as tf
from .layer import NetworkLayer


class DenseLayer(NetworkLayer):
    def __init__(self, network, index, batch_norm=None, hidden_sizes=[128], activation=tf.nn.relu,
                 weights_initializer=None, biases_initializer=None, weight_decay=None):
        super().__init__(network, index, batch_norm=batch_norm)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.weight_decay = weight_decay

    def build(self, inputs):
        # get variables
        self.dropout = self.context.get_or_create_feed("dropout")

        # initializers
        weights_initializer = self.load_initializer(self.weights_initializer)
        self.references["weights_initializer"] = weights_initializer
        biases_initializer = self.load_initializer(self.biases_initializer)
        self.references["weights_initializer"] = biases_initializer

        # prepare input
        input_size = np.prod(inputs.shape[1:])
        y = tf.reshape(tf.cast(inputs, tf.float32), (-1, input_size))

        # create fully connected layers
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

        return y

    def build_loss(self, targets):
        # evaluate continuous loss (MSE)
        loss = tf.reduce_mean(tf.square(targets - self.outputs))

        # evaluate accuracy
        accuracy = tf.exp(-loss)
        metrics = {"accuracy": accuracy}

        return loss, metrics

    def prepare_default_feeds(self, query, feed_map):
        feed_map["dropout"] = 1
        return feed_map
