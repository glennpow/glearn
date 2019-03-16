import tensorflow as tf
from .layer import NetworkLayer


class Conv2dLayer(NetworkLayer):
    def __init__(self, network, index, filters, input_shape=None, strides=1,
                 padding="SAME", activation=tf.nn.relu, lrn=None,
                 pooling="max", pool_k=2, pool_strides=2,
                 batch_norm=None, weights_initializer=None, biases_initializer=None):
        super().__init__(network, index, batch_norm=batch_norm)

        self.filters = filters
        self.input_shape = input_shape
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.lrn = lrn
        self.pooling = pooling
        self.pool_k = pool_k
        self.pool_strides = pool_strides
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs):
        # initializers
        weights_initializer = self.load_initializer(self.weights_initializer,
                                                    tf.contrib.layers.xavier_initializer())
        biases_initializer = self.load_initializer(self.biases_initializer,
                                                   tf.constant_initializer(0.0))

        # prepare input
        if self.input_shape is not None:
            x = tf.reshape(inputs, [-1] + self.input_shape)
        else:
            x = inputs
        x = tf.cast(x, tf.float32)
        input_channels = x.shape[-1]

        # create convolution layers
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

                    # batch normalization variables
                    self.prepare_batch_norm(output_channels)

                # conv2d and biases
                Z = tf.nn.conv2d(x, W, [1, self.strides, self.strides, 1], self.padding)
                Z = tf.nn.bias_add(Z, b)
                self.references["Z"] = Z

                # apply batch norm
                Z = self.apply_batch_norm(Z, [0, 1, 2])

                # activation
                if self.activation is not None:
                    activation_func = self.load_callable(self.activation)
                    A = activation_func(Z)
                else:
                    A = Z
                self.references["activation"] = A

                # local response normalization (before max pooling)
                lrn_order = None
                if self.lrn is not None:
                    lrn_order, lrn_bias, lrn_alpha, lrn_beta = self.lrn
                    if lrn_order:
                        A = tf.nn.lrn(A, bias=lrn_bias, alpha=lrn_alpha, beta=lrn_beta)

                # pooling
                if self.pooling is not None:
                    self.references["unpooled"] = A
                    ksize = [1, self.pool_k, self.pool_k, 1]
                    strides = [1, self.pool_strides, self.pool_strides, 1]
                    pooling_op = tf.nn.max_pool if self.pooling == "max" else tf.nn.avg_pool
                    A = pooling_op(Z, ksize=ksize, strides=strides, padding=self.padding)

                # local response normalization (after max pooling)
                if lrn_order is False:
                    A = tf.nn.lrn(A, bias=lrn_bias, alpha=lrn_alpha, beta=lrn_beta)

                features.append(A)
                x = A
                input_channels = output_channels
        self.references["features"] = features

        return x
