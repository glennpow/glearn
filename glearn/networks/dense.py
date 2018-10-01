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

    def build(self, inputs, outputs=None):
        # get variables
        dropout = self.context.get_or_create_feed("dropout")

        # initializers
        weights_initializer = self.load_initializer(self.weights_initializer)
        biases_initializer = self.load_initializer(self.biases_initializer)

        # create fully connected layers
        input_size = np.prod(inputs.shape[1:])
        x = tf.reshape(tf.cast(inputs, tf.float32), (-1, input_size))
        layers = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = self.dense(x, i, hidden_size, dropout, self.activation,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
            layers.append(x)
        self.references["layers"] = layers

        # if inference only, then return
        if outputs is None:
            return x

        # create output layer
        output_interface = outputs.interface
        i = len(self.hidden_sizes)
        if output_interface.discrete:
            y = self.dense(x, i, output_interface.size, dropout, tf.nn.softmax,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
        else:
            y = self.dense(x, i, output_interface.size, dropout, None,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)

        # loss evaluations
        with tf.name_scope('evaluate'):
            if output_interface.discrete:
                # evaluate discrete loss
                logits = self.references["Z"]
                neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=outputs)
                loss = tf.reduce_mean(neg_log_p)
                self.context.set_fetch("loss", loss, "evaluate")

                # evaluate accuracy
                correct = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                self.context.set_fetch("accuracy", accuracy, "evaluate")
            else:
                # evaluate continuous loss
                loss = tf.reduce_mean(tf.square(outputs - y))
                self.context.set_fetch("loss", loss, "evaluate")

                # evaluate accuracy
                accuracy = tf.exp(-loss)
                self.context.set_fetch("accuracy", accuracy, "evaluate")
        return y

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map
