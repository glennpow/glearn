import numpy as np
import tensorflow as tf
from .layer import NetworkLayer


class DenseLayer(NetworkLayer):
    def __init__(self, network, index, hidden_sizes=[128], activation=tf.nn.relu,
                 weights_initializer=None, biases_initializer=None):
        super().__init__(network, index)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

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
        x = tf.reshape(tf.cast(inputs, tf.float32), (-1, input_size))
        layers = []
        for hidden_size in self.hidden_sizes:
            x = self.dense(x, hidden_size, self.dropout, self.activation,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
            layers.append(x)
        self.references["layers"] = layers

        return x

    def build_predict(self, y):
        # create output layer
        output_interface = self.network.context.output
        if output_interface.discrete:
            y = self.dense(y, output_interface.size, self.dropout, tf.nn.softmax,
                           weights_initializer=self.references["weights_initializer"],
                           biases_initializer=self.references["weights_initializer"])
        else:
            y = self.dense(y, output_interface.size, self.dropout, None,
                           weights_initializer=self.references["weights_initializer"],
                           biases_initializer=self.references["weights_initializer"])
        return y

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map
