import tensorflow as tf
from .layer import NetworkLayer


class Conv2dLayer(NetworkLayer):
    def __init__(self, network, index, filters, input_channels=None, strides=1, padding="SAME",
                 activation=tf.nn.relu, max_pool_k=2, max_pool_strides=2, lrn=None,
                 weights_initializer=None, biases_initializer=None):
        super().__init__(network, index)

        self.filters = filters
        self.input_channels = input_channels
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.max_pool_k = max_pool_k
        self.max_pool_strides = max_pool_strides
        self.lrn = lrn
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs):
        # initializers
        weights_initializer = self.load_initializer(self.weights_initializer,
                                                    tf.contrib.layers.xavier_initializer())
        biases_initializer = self.load_initializer(self.biases_initializer,
                                                   tf.constant_initializer(0.0))

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
                    W = self.get_variable("W", (height, width, input_channels, output_channels),
                                          initializer=weights_initializer,
                                          trainable=self.trainable, cpu=True)
                    b = self.get_variable("b", (output_channels),
                                          initializer=biases_initializer,
                                          trainable=self.trainable, cpu=True)

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
                self.references["activation"] = A

                # local response normalization (before max pooling)
                lrn_order = None
                if self.lrn is not None:
                    lrn_order, lrn_bias, lrn_alpha, lrn_beta = self.lrn
                    if lrn_order:
                        A = tf.nn.lrn(A, bias=lrn_bias, alpha=lrn_alpha, beta=lrn_beta)

                # max pooling
                if self.max_pool_k is not None:
                    self.references["unpooled"] = A
                    ksize = [1, self.max_pool_k, self.max_pool_k, 1]
                    strides = [1, self.max_pool_strides, self.max_pool_strides, 1]
                    A = tf.nn.max_pool(Z, ksize=ksize, strides=strides, padding=self.padding)

                # local response normalization (after max pooling)
                if lrn_order is False:
                    A = tf.nn.lrn(A, bias=lrn_bias, alpha=lrn_alpha, beta=lrn_beta)

                features.append(A)
                x = A
                input_channels = output_channels
        self.references["features"] = features

        return x
