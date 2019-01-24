import numpy as np
import tensorflow as tf
from .distribution import DistributionLayer


class NormalDistributionLayer(DistributionLayer):
    def __init__(self, network, index, size=None,
                 mean_activation=tf.nn.tanh, mean_scale=1,
                 sigma_activation=tf.nn.softplus, sigma_scale=1,
                 weights_initializer=None, biases_initializer=None, l2_loss_coef=None, **kwargs):
        super().__init__(network, index, **kwargs)

        self.size = size or self.context.output.size
        self.mean_activation = mean_activation
        self.mean_scale = mean_scale
        self.sigma_activation = sigma_activation
        self.sigma_scale = sigma_scale
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.l2_loss_coef = l2_loss_coef

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
        mu = self.dense(x, self.size, dropout, self.mean_activation,
                        weights_initializer=weights_initializer,
                        biases_initializer=biases_initializer)
        if self.mean_scale != 1:
            mu *= self.mean_scale
        self.references["mu"] = mu
        self.summary.add_scalar("mu", tf.reduce_mean(mu), "evaluate")

        # mu L2 preactivation loss  (HACK?  this only works when |mu| should be = 0?)
        if self.l2_loss_coef is not None and self.l2_loss_coef > 0:
            # FIXME - may need axis=-1 here...
            l2_loss = tf.reduce_mean(tf.square(self.references["Z"])) * self.l2_loss_coef
            self.add_loss(l2_loss)
            self.summary.add_scalar("l2_loss", l2_loss, "evaluate")

        # create dense layer for sigma with scaling (no dropout to avoid div by zero)
        sigma = self.dense(x, self.size, None, self.sigma_activation,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
        if self.sigma_scale != 1:
            sigma *= self.sigma_scale
        self.references["sigma"] = sigma
        self.summary.add_scalar("sigma", tf.reduce_mean(sigma), "evaluate")

        # normal distribution
        distribution = tf.distributions.Normal(loc=mu, scale=sigma)
        self.references["distribution"] = distribution

        # sample from distribution
        y = distribution.sample()

        # TODO - more summaries
        # if hasattr(policy_distribution, "stddev"):
        #     action_stddev = tf.reduce_mean(tf.squeeze(policy_distribution.stddev()))
        #     self.summary.add_scalar("action_stddev", action_stddev, "evaluate")

        return y

    def prepare_default_feeds(self, queries, feed_map):
        feed_map["dropout"] = 1
        return feed_map
