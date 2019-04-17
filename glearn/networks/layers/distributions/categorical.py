import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from .distribution import DistributionLayer
from glearn.utils import tf_utils


class CategoricalDistributionLayer(DistributionLayer):
    def __init__(self, network, index, categories=None,
                 weights_initializer=None, biases_initializer=None, **kwargs):
        super().__init__(network, index, **kwargs)

        self.categories = categories
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs):
        self._build_categorical(inputs)

        y = self.references["category"]

        return y

    @property
    def probs(self):
        return self.distribution.probs

    def build_loss(self, targets):
        # evaluate discrete loss
        loss = tf.reduce_mean(self.neg_log_prob(targets))

        # evaluate accuracy
        correct = tf.equal(self.references["category"], targets)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        metrics = {"accuracy": accuracy}

        return loss, metrics

    def prepare_default_feeds(self, query, feed_map):
        feed_map["dropout"] = 1
        return feed_map

    def _build_categorical(self, inputs):
        # get variables
        dropout = self.context.get_or_create_feed("dropout")

        # create dense layer for logits
        x = tf_utils.flatten(inputs, axis=1)
        if not isinstance(self.categories, int):
            self.categories = self.context.output.size
        logits = self.dense(x, self.categories, dropout, None,
                            weights_initializer=self.weights_initializer,
                            biases_initializer=self.biases_initializer)

        # categorical distribution
        self.references["logits"] = logits
        distribution = tf.distributions.Categorical(logits=logits)
        self.references["distribution"] = distribution

        # sample from distribution
        if self.context.output.deterministic:
            y = tf.argmax(distribution.probs, -1, name="sample", output_type=tf.int32)
        else:
            y = self.sample(name="sample")
        self.references["category"] = y

        return distribution


class DiscretizedDistributionLayer(CategoricalDistributionLayer):
    def __init__(self, network, index, divs, low=0, high=1,
                 weights_initializer=None, biases_initializer=None, **kwargs):
        super().__init__(network, index, categories=divs,
                         weights_initializer=weights_initializer,
                         biases_initializer=biases_initializer, **kwargs)
        self.divs = divs
        self.low = low
        self.high = high

    def build(self, inputs):
        # get categorical outputs
        distribution = self._build_categorical(inputs)

        # wrap categorical outputs in bijector which converts into discretized range
        self.bijector = DiscretizedBijector(self.divs, self.low, self.high)
        distribution = tf.contrib.distributions.TransformedDistribution(distribution=distribution,
                                                                        bijector=self.bijector,
                                                                        name="discretized")
        self.references["distribution"] = distribution

        # sample from distribution
        y = tf.cast(distribution.sample(), tf.float32)

        return y

    @property
    def probs(self):
        return self.distribution.distribution.probs

    def neg_log_prob(self, value, **kwargs):
        value = self.bijector._inverse(value)
        return super().neg_log_prob(value, **kwargs)


class DiscretizedBijector(tfd.bijectors.Bijector):
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
        return tf.cast((y - self.low) / (self.high - self.low) * (self.divs - 1), tf.int32)
        # return (y - self.low) / (self.high - self.low) * (self.divs - 1)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype.base_dtype)
