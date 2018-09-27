import numpy as np
import tensorflow as tf
from .layer import NetworkLayer


class FullyConnectedLayer(NetworkLayer):
    def __init__(self, network, index, hidden_sizes=[128], activation=tf.nn.relu,
                 initializer=None):
        super().__init__(network, index)

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.initializer = initializer

    def build(self, inputs, outputs=None):
        # get variables
        dropout = self.context.get_feed("dropout")
        if dropout is None:
            dropout = tf.placeholder(tf.float32, (), name="dropout")
            self.context.set_feed("dropout", dropout)

        # initializer
        initializer_seed = 1
        if isinstance(self.initializer, int):
            initializer_seed = self.initializer
            self.initializer = None
        if self.initializer is None:
            self.initializer = tf.contrib.layers.xavier_initializer(seed=initializer_seed)

        # create fully connected layers
        input_size = np.prod(inputs.shape[1:])
        x = tf.reshape(tf.cast(inputs, tf.float32), (-1, input_size))
        layers = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = self.fully_connected(x, i, hidden_size, dropout, self.activation)
            layers.append(x)
        self.references["layers"] = layers

        # if inference only, then return
        if outputs is None:
            return x

        # create output layer
        output_interface = outputs.interface
        i = len(self.hidden_sizes)
        if output_interface.discrete:
            y = self.fully_connected(x, i, output_interface.size, dropout, tf.nn.softmax)
        else:
            y = self.fully_connected(x, i, output_interface.size, dropout, None)

        # loss evaluations
        with tf.name_scope('evaluate'):
            if output_interface.discrete:
                # evaluate loss
                logits = self.references["Z"]
                neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=outputs)
                loss = tf.reduce_mean(neg_log_p)
                self.context.set_fetch("loss", loss, "evaluate")

                # TODO - stochastic discrete also
                correct = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                self.context.set_fetch("accuracy", accuracy, "evaluate")
            else:
                loss = tf.reduce_mean(tf.square(outputs - y))
                self.context.set_fetch("loss", loss, "evaluate")
        return y

    def fully_connected(self, x, index, hidden_size, dropout, activation):
        scope = f"fc_{self.index}_{index}"
        with tf.name_scope(scope):
            # create variables
            with tf.variable_scope(scope):
                W = tf.get_variable("W", (x.shape[1], hidden_size),
                                    initializer=self.initializer)
                b = tf.get_variable("b", (hidden_size, ), initializer=self.initializer)

            # weights and biases
            Z = tf.matmul(x, W)
            Z = tf.add(Z, b)

            # activation
            if activation is not None:
                self.references["Z"] = Z
                A = activation(Z)
            else:
                A = Z

            # dropout
            if dropout is not None:
                self.references["undropped"] = A
                A = tf.nn.dropout(A, dropout)
        return A

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map
