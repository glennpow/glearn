import numpy as np
import tensorflow as tf
from .layer import NetworkLayer


class DistributionLayer(NetworkLayer):
    def __init__(self, network, index, distribution="normal",
                 mean_activation=tf.nn.tanh, mean_scale=1,
                 sigma_activation=tf.nn.softplus, sigma_scale=1,
                 weights_initializer=None, biases_initializer=None):
        super().__init__(network, index)

        self.mean_activation = mean_activation
        self.mean_scale = mean_scale
        self.sigma_activation = sigma_activation
        self.sigma_scale = sigma_scale
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer

    def build(self, inputs, outputs=None):
        # get variables
        dropout = self.context.get_or_create_feed("dropout")

        # initializer
        weights_initializer = self.load_initializer(self.weights_initializer)
        biases_initializer = self.load_initializer(self.biases_initializer)

        # create dense layer for mu
        input_size = np.prod(inputs.shape[1:])
        x = tf.reshape(inputs, (-1, input_size))
        output_size = self.context.output.size
        mu = self.dense(x, 0, output_size, dropout, self.mean_activation,
                        weights_initializer=weights_initializer,
                        biases_initializer=biases_initializer)
        if self.mean_scale != 1:
            mu *= self.mean_scale
        self.references["mu"] = mu

        # create dense layer for sigma
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

        # if inference only, then return
        if outputs is None:
            return x

        # TODO - could extract loss components from layers, and share them
        raise Exception("No evaluation logic available for DistributionLayer")

    def prepare_default_feeds(self, graphs, feed_map):
        feed_map["dropout"] = 1
        return feed_map

    def prob(self, output):
        return self.references["distribution"].prob(output)

    def log_prob(self, output):
        return self.references["distribution"].log_prob(output)
