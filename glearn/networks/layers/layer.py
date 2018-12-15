import tensorflow as tf
from glearn.utils.reflection import get_class, get_function


def load_layer(network, index, definition):
    LayerClass = get_class(definition)

    return LayerClass(network, index)


class NetworkLayer(object):
    def __init__(self, network, index):
        self.network = network
        self.index = index
        self.dense_count = 0
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

    def load_initializer(self, definition=None, default=None):
        if definition is None:
            # use default initializer
            return default
        else:
            # check if initializer already loaded
            if callable(definition):
                return definition

            # load initializer constructor and call
            initializer_function = get_function(definition)
            return initializer_function()

    def get_variable(self, name, shape, **kwargs):
        return tf.get_variable(name, shape=shape, **kwargs)

    def add_loss(self, loss):
        self.network.add_loss(loss)

    def build(self, inputs):
        # override
        return inputs

    def build_predict(self, y):
        # override
        return y

    def prepare_default_feeds(self, graphs, feed_map):
        # override
        return feed_map

    def dense(self, x, hidden_size, dropout=None, activation=None,
              weights_initializer=None, biases_initializer=None, weight_decay=None):
        # create common single dense layer
        dense_index = self.dense_count
        self.dense_count += 1
        scope = f"dense_{self.index}_{dense_index}"
        with tf.name_scope(scope):
            # create variables
            with tf.variable_scope(scope):
                # weights
                weights_initializer = self.load_initializer(weights_initializer,
                                                            tf.contrib.layers.xavier_initializer())
                W = self.get_variable("W", (x.shape[1], hidden_size),
                                      initializer=weights_initializer, trainable=self.trainable)

                # weight decay loss
                if weight_decay is not None:
                    W_loss = tf.multiply(tf.nn.l2_loss(W), weight_decay, name='W_loss')
                    self.context.set_fetch(f"{scope}_W_loss", W_loss, ["evaluate"])
                    self.add_loss(W_loss)

                # biases
                biases_initializer = self.load_initializer(biases_initializer,
                                                           tf.constant_initializer(0.0))
                b = self.get_variable("b", (hidden_size, ), initializer=biases_initializer,
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
