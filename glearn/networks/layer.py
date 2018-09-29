import tensorflow as tf
from glearn.utils.reflection import get_class, get_function


def load_layer(network, index, definition):
    LayerClass = get_class(definition)

    return LayerClass(network, index)


class NetworkLayer(object):
    def __init__(self, network, index):
        self.network = network
        self.index = index
        self.references = {}

    @property
    def layer_type(self):
        return type(self).__name__

    @property
    def context(self):
        return self.network.context

    @property
    def trainable(self):
        return self.network.trainable

    @property
    def seed(self):
        return self.network.seed

    def load_initializer(self, definition=None):
        if definition is None:
            return tf.contrib.layers.xavier_initializer(seed=self.seed)
        else:
            initializer_function = get_function(definition)
            return initializer_function(seed=self.seed)

    def build(self, inputs, outputs=None):
        pass

    def prepare_default_feeds(self, graphs, feed_map):
        return feed_map

    def dense(self, x, index, hidden_size, dropout=None, activation=None, initializer=None):
        # create common single dense layer
        scope = f"dense_{self.index}_{index}"
        with tf.name_scope(scope):
            # create variables
            with tf.variable_scope(scope):
                W = tf.get_variable("W", (x.shape[1], hidden_size),
                                    initializer=initializer, trainable=self.trainable)
                b = tf.get_variable("b", (hidden_size, ), initializer=initializer,
                                    trainable=self.trainable)

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
