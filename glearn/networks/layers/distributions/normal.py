import numpy as np
import tensorflow as tf
# import tensorflow.contrib.distributions as tfd
from .distribution import DistributionLayer


EPSILON = 1e-6


class NormalDistributionLayer(DistributionLayer):
    def __init__(self, network, index, size=None,
                 mu_activation=tf.nn.tanh, mu_scale=1,
                 sigma_activation=tf.nn.softplus, sigma_scale=1, squash=False,
                 weights_initializer=None, biases_initializer=None, l2_loss_coef=None, **kwargs):
        super().__init__(network, index, **kwargs)

        self.size = size or self.context.output.size
        self.mu_activation = mu_activation
        self.mu_scale = mu_scale
        self.sigma_activation = sigma_activation
        self.sigma_scale = sigma_scale
        self.squash = squash
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.l2_loss_coef = l2_loss_coef

    def build(self, inputs):
        # get variables
        dropout = self.context.get_or_create_feed("dropout")

        # create dense layer for mu
        input_size = np.prod(inputs.shape[1:])
        x = tf.reshape(inputs, (-1, input_size))
        self.mu = self.dense(x, self.size, dropout, self.mu_activation,
                             weights_initializer=self.weights_initializer,
                             biases_initializer=self.biases_initializer)
        if self.mu_scale != 1:
            self.mu *= self.mu_scale
        self.references["mu"] = self.mu
        if self.size == 1:
            self.summary.add_histogram("mu", self.mu)
        else:
            for i in range(self.size):
                self.summary.add_histogram(f"mu_{i}", self.mu[:, i])

        # mean L2 preactivation loss  (HACK?  this only works when |mu| should be = 0?)
        if self.l2_loss_coef is not None and self.l2_loss_coef > 0:
            # FIXME - may need axis=-1 here...
            l2_loss = tf.reduce_mean(tf.square(self.references["Z"])) * self.l2_loss_coef
            self.add_loss(l2_loss)
            self.summary.add_scalar("l2_loss", l2_loss)

        # create dense layer for sigma with scaling
        self.sigma = self.dense(x, self.size, dropout, self.sigma_activation,
                                weights_initializer=self.weights_initializer,
                                biases_initializer=self.biases_initializer)
        if self.sigma_scale != 1:
            self.sigma *= self.sigma_scale
        self.sigma += 1e-6
        self.references["sigma"] = self.sigma
        if self.size == 1:
            self.summary.add_histogram("sigma", self.sigma)
        else:
            for i in range(self.size):
                self.summary.add_histogram(f"sigma_{i}", self.sigma[:, i])

        # normal distribution
        distribution = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
        self.references["distribution"] = distribution

        # tanh squashing (FIXME - this didn't work)
        # if self.squash:
        #     self.bijector = TanhBijector()
        #     distribution = tfd.TransformedDistribution(distribution=distribution,
        #                                                bijector=self.bijector,
        #                                                name="tanh")
        #     self.references["distribution"] = distribution

        # sample from distribution
        y = self.sample()

        # log prediction distribution
        self.summary.add_histogram("predict", y)

        return y

    def build_loss(self, targets):
        # evaluate continuous loss  (FIXME?)
        loss = tf.reduce_mean(self.neg_log_prob(targets))

        # evaluate accuracy
        correct = tf.exp(-tf.abs(self.outputs - targets))
        accuracy = tf.reduce_mean(correct)
        metrics = {"accuracy": accuracy}

        return loss, metrics

    def prepare_default_feeds(self, query, feed_map):
        feed_map["dropout"] = 1
        return feed_map

    # FIXME - these functions below are required until tensorflow upgrade :-/

    def prob(self, value, **kwargs):
        if self.squash:
            value = tf.atanh(value)
        prob = super().prob(value, **kwargs)
        prob = tf.reduce_prod(prob, axis=-1)
        return prob

    def log_prob(self, value, **kwargs):
        if self.squash:
            # from SAC paper: https://arxiv.org/pdf/1801.01290.pdf
            u = tf.atanh(value)
            correction = tf.reduce_sum(tf.log(1 - value ** 2 + EPSILON), axis=1)
            # correction = tf.reduce_sum(tf.log1p(-tf.square(value) + EPSILON), axis=1)
            log_prob = super().log_prob(u, **kwargs) - correction
        else:
            log_prob = super().log_prob(value, **kwargs)
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        return log_prob

    def mean(self, **kwargs):
        y = super().mean(**kwargs)
        if self.squash:
            y = tf.tanh(y)
        return y

    def mode(self, **kwargs):
        y = super().mode(**kwargs)
        if self.squash:
            y = tf.tanh(y)
        return y

    def sample(self, **kwargs):
        y = super().sample(**kwargs)
        if self.squash:
            return tf.tanh(y)
        return y


# class TanhBijector(tfd.bijectors.Bijector):
#     def __init__(self, validate_args=False, name="tanh"):
#         super().__init__(validate_args=validate_args, forward_min_event_ndims=0, name=name)

#     def _forward(self, x):
#         return tf.tanh(x)

#     def _inverse(self, y):
#         return tf.atanh(y)

#     def _inverse_log_det_jacobian(self, y):
#         return -tf.log1p(-tf.square(y) + EPSILON)

#     def _forward_log_det_jacobian(self, x):
#         return 2. * (np.log(2.) - x - tf.nn.softplus(-2. * x))
