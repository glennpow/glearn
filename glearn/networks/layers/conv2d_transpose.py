import numpy as np
import tensorflow as tf
from .layer import NetworkLayer


class Conv2dTransposeLayer(NetworkLayer):
    def __init__(self, network, index, filters, input_shape=None, strides=1,
                 padding="SAME", activation=tf.nn.relu, batch_norm=None, weights_initializer=None,
                 biases_initializer=None):
        super().__init__(network, index, batch_norm=batch_norm)

        self.filters = filters
        self.input_shape = input_shape
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def calculate_output_shape(self, inputs, output_channels):
        # FIXME - make work for all cases
        input_shape = inputs.shape
        return [tf.shape(inputs)[0], self.strides * input_shape[1], self.strides * input_shape[2],
                output_channels]

    def build(self, inputs):
        # initializers
        weights_initializer = self.load_initializer(self.weights_initializer,
                                                    tf.contrib.layers.xavier_initializer())
        biases_initializer = self.load_initializer(self.biases_initializer,
                                                   tf.constant_initializer(0.0))

        # prepare inputs
        if self.input_shape is not None:
            # TODO - could verify input shape doesn't already match
            hidden_size = np.prod(self.input_shape)
            x = self.dense(inputs, hidden_size, weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
            x = tf.reshape(x, [-1] + self.input_shape)
        else:
            x = inputs
        x = tf.cast(x, tf.float32)
        input_channels = x.shape[-1]

        # create convolution layers
        features = []
        for i, filter in enumerate(self.filters):
            scope = f"deconv2d_{self.index}_{i}"
            with tf.name_scope(scope):
                with tf.variable_scope(scope):
                    # create variables
                    height, width, output_channels = filter
                    W = self.get_variable("W", (height, width, output_channels, input_channels),
                                          initializer=weights_initializer,
                                          trainable=self.trainable, cpu=True)
                    b = self.get_variable("b", (output_channels),
                                          initializer=biases_initializer,
                                          trainable=self.trainable, cpu=True)

                    # batch normalization variables
                    self.prepare_batch_norm(output_channels)

                # conv2d and biases
                output_shape = self.calculate_output_shape(x, output_channels)
                Z = tf.nn.conv2d_transpose(x, W, output_shape, [1, self.strides, self.strides, 1],
                                           padding=self.padding)
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

                features.append(A)
                x = A
                input_channels = output_channels
        self.references["features"] = features

        return x
