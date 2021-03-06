import tensorflow as tf
from glearn.utils.reflection import get_class, get_function
from glearn.utils.device import get_device


def load_layer(network, index, definition):
    LayerClass = get_class(definition)

    return LayerClass(network, index)


class NetworkLayer(object):
    def __init__(self, network, index, batch_norm=None):  # TODO - weight decay in here too
        self.network = network
        self.index = index
        self.batch_norm = batch_norm

        self.dense_count = 0
        self.references = {}

    @property
    def layer_type(self):
        return type(self).__name__

    @property
    def config(self):
        return self.network.config

    @property
    def context(self):
        return self.network.context

    @property
    def summary(self):
        return self.network.summary

    @property
    def trainable(self):
        return self.network.trainable

    @property
    def seed(self):
        return self.network.seed

    def load_callable(self, definition=None, default=None, call=False):
        if definition is None:
            # use default
            return default
        else:
            # check if already a callable
            if callable(definition):
                return definition

            # load function and call
            func = get_function(definition)
            if call:
                func = func()
            return func

    def load_initializer(self, definition=None, default=None):
        return self.load_callable(definition, default, call=True)

    def get_variable(self, name, shape, cpu=None, gpu=None, **kwargs):
        device = get_device(cpu=cpu, gpu=gpu)
        if device is not None:
            with tf.device(device):
                return tf.get_variable(name, shape=shape, **kwargs)
        else:
            return tf.get_variable(name, shape=shape, **kwargs)

    def uses_batch_norm(self, override=None):
        if override is None:
            override = self.batch_norm
        if override is None:
            override = self.network.definition.get("batch_norm", False)
        return override

    def prepare_batch_norm(self, size, override=None):
        # batch normalization variables
        if self.uses_batch_norm(override):
            offset = self.get_variable("offset", [size], initializer=tf.zeros_initializer())
            self.references["batch_norm_offset"] = offset
            scale = self.get_variable("scale", [size], initializer=tf.ones_initializer())
            self.references["batch_norm_scale"] = scale

    def apply_batch_norm(self, Z, axes=[0], override=None):
        # apply batch normalization
        if self.uses_batch_norm(override):
            with tf.variable_scope("batch_norm"):
                mean, var = tf.nn.moments(Z, axes)
                offset = self.references["batch_norm_offset"]
                scale = self.references["batch_norm_scale"]
                epsilon = 1e-3
                return tf.nn.batch_normalization(Z, mean, var, offset, scale, epsilon)
        return Z

    def add_loss(self, loss):
        self.network.add_loss(loss)

    def variable_scope(self, name_or_scope, **kwargs):
        return self.context.variable_scope(name_or_scope, **kwargs)

    def build(self, inputs):
        # override
        return inputs

    def build_predict(self, y):
        # override
        return y

    def build_loss(self, targets):
        # override
        return tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)

    def prepare_default_feeds(self, query, feed_map):
        # override
        return feed_map

    def dense(self, x, hidden_size, dropout=None, activation=None, batch_norm=None,
              weights_initializer=None, biases_initializer=None, weight_decay=None):
        # create common single dense layer
        dense_index = self.dense_count
        self.dense_count += 1
        scope = f"dense_{self.index}_{dense_index}"
        with tf.variable_scope(scope):
            # prepare inputs
            if len(x.shape) == 1:
                x = tf.expand_dims(x, axis=-1)

            # weights
            weights_initializer = self.load_initializer(weights_initializer,
                                                        tf.contrib.layers.xavier_initializer())
            W = self.get_variable("W", (x.shape[1], hidden_size), cpu=True,
                                  initializer=weights_initializer, trainable=self.trainable)

            # weight decay loss
            if weight_decay is not None:
                W_loss = tf.multiply(tf.nn.l2_loss(W), weight_decay, name='W_loss')
                self.context.add_fetch(f"{scope}_W_loss", W_loss, ["evaluate"])
                self.add_loss(W_loss)

            # biases
            biases_initializer = self.load_initializer(biases_initializer,
                                                       tf.constant_initializer(0.0))
            b = self.get_variable("b", (hidden_size, ), cpu=True,
                                  initializer=biases_initializer, trainable=self.trainable)

            # batch normalization variables
            self.prepare_batch_norm(hidden_size, batch_norm)

            # weights and biases
            Z = tf.matmul(x, W)
            Z = tf.add(Z, b)
            self.references["Z"] = Z

            # apply batch normalization
            Z = self.apply_batch_norm(Z, [0], batch_norm)

            # activation
            if activation is not None:
                activation_func = self.load_callable(activation)
                A = activation_func(Z)
            else:
                A = Z
            self.references["activation"] = A

            # dropout
            if dropout is not None:
                self.references["undropped"] = A
                A = tf.nn.dropout(A, dropout)
        return A

    def activation_summary(self, query=None):
        activation = self.references.get("activation")
        if activation is not None:
            self.summary.add_activation(activation, query=query)
