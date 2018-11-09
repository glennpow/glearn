import numpy as np
import tensorflow as tf
from .distribution import DistributionLayer


class CategoricalDistributionLayer(DistributionLayer):
    def __init__(self, network, index, categories=None,
                 weights_initializer=None, biases_initializer=None):
        super().__init__(network, index)

        self.categories = categories
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs):
        # get variables
        dropout = self.context.get_or_create_feed("dropout")

        # initializer
        weights_initializer = self.load_initializer(self.weights_initializer,
                                                    default=tf.contrib.layers.xavier_initializer())
        biases_initializer = self.load_initializer(self.biases_initializer,
                                                   default=tf.contrib.layers.xavier_initializer())

        # create dense layer for mu
        input_size = np.prod(inputs.shape[1:])
        x = tf.reshape(inputs, (-1, input_size))
        output_size = self.categories
        if not isinstance(output_size, int):
            output_size = self.context.output.size
        x = self.dense(x, output_size, dropout, None,
                       weights_initializer=weights_initializer,
                       biases_initializer=biases_initializer)

        # categorical distribution
        self.references["logits"] = x
        x = tf.distributions.Categorical(logits=x)
        self.references["distribution"] = x

        # sample from distribution
        x = x.sample(1, seed=self.seed)
        # x = tf.squeeze(x, axis=0)  # TODO FIXME?

        return x

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map


class DiscretizedDistributionLayer(CategoricalDistributionLayer):
    def __init__(self, network, index, divs, low=0, high=1,
                 weights_initializer=None, biases_initializer=None):
        super().__init__(network, index, categories=divs,
                         weights_initializer=weights_initializer,
                         biases_initializer=biases_initializer)
        self.divs = divs
        self.low = low
        self.high = high

    def build(self, inputs):
        # get categorical outputs
        x = super().build(inputs)

        # wrap categorical outputs in bijector which converts into discretized range
        self.bijector = DiscretizedBijector(self.divs, self.low, self.high)
        x = tf.contrib.distributions.TransformedDistribution(distribution=self.distribution,
                                                             bijector=self.bijector)
        self.references["distribution"] = x

        # return sample
        x = x.sample(1, seed=self.seed)

        return x


class DiscretizedBijector(tf.contrib.distributions.bijectors.Bijector):
    def __init__(self, divs, low=0, high=1, validate_args=False, name="discretized"):
        super().__init__(validate_args=validate_args, forward_min_event_ndims=0,
                         is_constant_jacobian=True, name=name)
        self.divs = divs
        self.low = low
        self.high = high

    def _forward(self, x):
        # convert categorical into discretized range
        return (x / (self.divs - 1)) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # convert discretized range into categorical
        # return tf.cast((y - self.low) / (self.high - self.low), tf.int32)
        return (y - self.low) / (self.high - self.low)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype.base_dtype)
