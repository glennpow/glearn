import numpy as np
import tensorflow as tf
from .distribution import DistributionLayer


class NormalDistributionLayer(DistributionLayer):
    def __init__(self, network, index,
                 mean_activation=tf.nn.tanh, mean_scale=1,
                 sigma_activation=tf.nn.softplus, sigma_scale=1,
                 weights_initializer=None, biases_initializer=None, l2_loss_coef=None):
        super().__init__(network, index)

        self.mean_activation = mean_activation
        self.mean_scale = mean_scale
        self.sigma_activation = sigma_activation
        self.sigma_scale = sigma_scale
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.l2_loss_coef = l2_loss_coef

    def build(self, inputs, outputs=None):
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
        output_size = self.context.output.size
        mu = self.dense(x, 0, output_size, dropout, None,
                        weights_initializer=weights_initializer,
                        biases_initializer=biases_initializer)

        # mu L2 preactivation loss  (HACK?  this only works when |mu| should be = 0?)
        if self.l2_loss_coef is not None and self.l2_loss_coef > 0:
            l2_loss = tf.reduce_mean(tf.square(mu)) * self.l2_loss_coef
            self.references["l2_loss"] = l2_loss

        # mu activation and scaling
        mu = self.mean_activation(mu)
        if self.mean_scale != 1:
            mu *= self.mean_scale
        self.references["mu"] = mu

        # create dense layer for sigma with scaling
        sigma = self.dense(x, 1, output_size, dropout, self.sigma_activation,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
        if self.sigma_scale != 1:
            sigma *= self.sigma_scale
        self.references["sigma"] = sigma

        # normal distribution
        x = tf.distributions.Normal(loc=mu, scale=sigma)
        self.references["distribution"] = x

        # sample from distribution
        x = x.sample(1, seed=self.seed)
        x = tf.squeeze(x, axis=0)

        # action clipping
        # x = tf.clip_by_value(x, -2, 2)  # HACK

        # if inference only, then return
        if outputs is None:
            return x

        # kl_divergence = tf.reduce_sum(distributions.kl(q_z, self.distribution), 1) ??

        # TODO - could extract loss components from layers, and share them
        raise Exception("No evaluation logic available for NormalDistributionLayer")

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map