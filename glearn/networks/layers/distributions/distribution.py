import tensorflow as tf
from glearn.networks.layers.layer import NetworkLayer


class DistributionLayer(NetworkLayer):
    @property
    def distribution(self):
        return self.references["distribution"]

    @property
    def logits(self):
        return self.distribution.logits

    def covariance(self, **kwargs):
        return self.distribution.covariance(**kwargs)

    def cross_entropy(self, other, **kwargs):
        return self.distribution.cross_entropy(other, **kwargs)

    def entropy(self, **kwargs):
        return self.distribution.entropy(**kwargs)

    def kl_divergence(self, other, **kwargs):
        return self.distribution.kl_divergence(other, **kwargs)

    def reshaped_targets(self, targets):
        return tf.squeeze(targets, axis=-1, name="reshape_targets")

    def prob(self, targets, name="FIXME", **kwargs):
        with tf.name_scope(name):  # FIXME - some sort of NOOP context if name=None
            return self.distribution.prob(self.reshaped_targets(targets), **kwargs)

    def log_prob(self, targets, name="FIXME", **kwargs):
        with tf.name_scope(name):  # FIXME - some sort of NOOP context if name=None
            return self.distribution.log_prob(self.reshaped_targets(targets), **kwargs)

    def neg_log_prob(self, targets, name="FIXME", **kwargs):
        with tf.name_scope(name):  # FIXME - some sort of NOOP context if name=None
            return -self.distribution.log_prob(self.reshaped_targets(targets), **kwargs)

    def mean(self, **kwargs):
        return self.distribution.mean(**kwargs)

    def mode(self, **kwargs):
        return self.distribution.mode(**kwargs)

    def sample(self, sample_shape=(), **kwargs):
        return self.distribution.sample(sample_shape=sample_shape, **kwargs)

    def stddev(self, **kwargs):
        return self.distribution.stddev(**kwargs)

    def variance(self, **kwargs):
        return self.distribution.variance(**kwargs)
