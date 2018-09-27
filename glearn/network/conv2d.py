import tensorflow as tf
from .layer import NetworkLayer


class Conv2dLayer(NetworkLayer):
    def __init__(self, index, filters, input_channels=None, strides=1, max_pool_k=2,
                 padding="SAME", activation=tf.nn.relu, initializer=None):
        super().__init__(index)

        self.filters = filters
        self.input_channels = input_channels
        self.strides = strides
        self.max_pool_k = max_pool_k
        self.padding = padding
        self.activation = activation
        self.initializer = initializer

    def build(self, policy, inputs, outputs=None):
        # get variables
        dropout = policy.get_feed("dropout")
        if dropout is None:
            dropout = tf.placeholder(tf.float32, (), name="dropout")
            policy.set_feed("dropout", dropout)

        # initializer
        initializer_seed = 1
        if isinstance(self.initializer, int):
            initializer_seed = self.initializer
            self.initializer = None
        if self.initializer is None:
            self.initializer = tf.contrib.layers.xavier_initializer(seed=initializer_seed)

        # create convolution layers
        input_channels = self.input_channels
        if input_channels is None:
            input_channels = inputs.shape[-1]
        # x = tf.reshape(inputs, shape=[-1, 28, 28, 1])  # TODO - pass/infer dimensions arg?
        x = tf.cast(inputs, tf.float32)
        features = []
        for i, filter in enumerate(self.filters):
            scope = f"conv2d_{self.index}_{i}"
            with tf.name_scope(scope):
                with tf.variable_scope(scope):
                    # create variables
                    height, width, output_channels = filter
                    W = tf.get_variable("W", (height, width, input_channels, output_channels),
                                        initializer=self.initializer)
                    b = tf.get_variable("b", (output_channels), initializer=self.initializer)

                # conv2d and biases
                Z = tf.nn.conv2d(x, W, strides=[1, self.strides, self.strides, 1],
                                 padding=self.padding)
                Z = tf.nn.bias_add(Z, b)

                # activation
                if self.activation is not None:
                    self.references["Z"] = Z
                    A = self.activation(Z)
                else:
                    A = Z

                # max pooling
                if self.max_pool_k is not None:
                    self.references["unpooled"] = A
                    ksize = [1, self.max_pool_k, self.max_pool_k, 1]
                    A = tf.nn.max_pool(Z, ksize=ksize, strides=ksize, padding=self.padding)

                features.append(A)
                x = A
                input_channels = output_channels
        self.references["features"] = features

        if outputs is None:
            return x

        raise Exception("No evaluation logic available for CNN")

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map
