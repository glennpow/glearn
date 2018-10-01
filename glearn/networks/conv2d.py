import tensorflow as tf
from .layer import NetworkLayer


class Conv2dLayer(NetworkLayer):
    def __init__(self, network, index, filters, input_channels=None, strides=1, max_pool_k=2,
                 padding="SAME", activation=tf.nn.relu,
                 weights_initializer=None,
                 biases_initializer=None):
        super().__init__(network, index)

        self.filters = filters
        self.input_channels = input_channels
        self.strides = strides
        self.max_pool_k = max_pool_k
        self.padding = padding
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs, outputs=None):
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
                    W = tf.get_variable("W", (height, width, input_channels, output_channels),
                                        initializer=weights_initializer,
                                        trainable=self.trainable)
                    b = tf.get_variable("b", (output_channels),
                                        initializer=biases_initializer,
                                        trainable=self.trainable)

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

        # TODO - could extract loss components from layers, and share them
        raise Exception("No evaluation logic available for Conv2dLayer")
