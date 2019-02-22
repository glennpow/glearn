import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from .distribution import DistributionLayer


class CategoricalDistributionLayer(DistributionLayer):
    def __init__(self, network, index, categories=None,
                 weights_initializer=None, biases_initializer=None, **kwargs):
        super().__init__(network, index, **kwargs)

        self.categories = categories
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs):
        self._build_categorical(inputs)

        y = tf.one_hot(self.references["category"], self.categories)

        return y

    def build_predict(self, y):
        y = super().build_predict(y)

        # log prediction distribution
        self.summary.add_histogram("category", self.references["category"])

        return y

    def build_loss(self, labels):
        # evaluate discrete loss
        loss = tf.reduce_mean(self.neg_log_prob(labels))

        # evaluate accuracy
        correct = tf.equal(self.references["category"], self.encipher(labels))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        return loss, accuracy

    def prepare_default_feeds(self, queries, feed_map):
        feed_map["dropout"] = 1
        return feed_map

    def encipher(self, one_hot):
        return tf.argmax(one_hot, axis=-1, output_type=tf.int32)

    def neg_log_prob(self, value, **kwargs):
        # NOTE: unfortunately, using -self.log_prob(value) does not return correct results
        logits = self.references["logits"]
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=value)

    def log_prob(self, value, **kwargs):
        # convert from one-hot
        return self.distribution.log_prob(self.encipher(value))

    def prob(self, value, **kwargs):
        # convert from one-hot
        return self.distribution.prob(self.encipher(value), **kwargs)

    def _build_categorical(self, inputs):
        # get variables
        dropout = self.context.get_or_create_feed("dropout")

        # initializer
        weights_initializer = self.load_initializer(self.weights_initializer,
                                                    default=tf.contrib.layers.xavier_initializer())
        biases_initializer = self.load_initializer(self.biases_initializer,
                                                   default=tf.contrib.layers.xavier_initializer())

        # create dense layer for logits
        input_size = np.prod(inputs.shape[1:])
        x = tf.reshape(inputs, (-1, input_size))
        if not isinstance(self.categories, int):
            self.categories = self.context.output.size
        logits = self.dense(x, self.categories, dropout, None,
                            weights_initializer=weights_initializer,
                            biases_initializer=biases_initializer)

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
        y = tf.expand_dims(tf.cast(distribution.sample(), tf.float32), -1)

        return y

    @property
    def probs(self):
        return self.distribution.distribution.probs


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
        # return tf.cast((y - self.low) / (self.high - self.low) * (self.divs - 1), tf.int32)
        return (y - self.low) / (self.high - self.low) * (self.divs - 1)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., x.dtype.base_dtype)
