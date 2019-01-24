import tensorflow as tf
from glearn.networks.layers.layer import NetworkLayer


class DistributionLayer(NetworkLayer):
    def __init__(self, network, index, ent_coef=None):
        super().__init__(network, index)

        self.ent_coef = ent_coef

    def build_predict(self, y):
        y = super().build_predict(y)

        # entropy exploration factor
        if self.config.reinforcement and self.ent_coef is not None and self.ent_coef > 0:
            entropy = tf.reduce_mean(self.distribution.entropy())
            self.summary.add_scalar("entropy", entropy, "evaluate")

            entropy_loss = -self.ent_coef * entropy
            self.add_loss(entropy_loss)
            self.summary.add_scalar("entropy_loss", entropy_loss, "evaluate")

        return y

    @property
    def distribution(self):
        return self.references["distribution"]

    @property
    def logits(self):
        return self.distribution.logits

    @property
    def probs(self):
        return self.distribution.probs

    def covariance(self, **kwargs):
        return self.distribution.covariance(**kwargs)

    def cross_entropy(self, other, **kwargs):
        return self.distribution.cross_entropy(other, **kwargs)

    def entropy(self, **kwargs):
        return self.distribution.entropy(**kwargs)

    def kl_divergence(self, other, **kwargs):
        return self.distribution.kl_divergence(other, **kwargs)

    def neg_log_prob(self, value, **kwargs):
        return -self.log_prob(value, **kwargs)

    def log_prob(self, value, **kwargs):
        return self.distribution.log_prob(value, **kwargs)

    def mean(self, **kwargs):
        return self.distribution.mean(**kwargs)

    def mode(self, **kwargs):
        return self.distribution.mode(**kwargs)

    def prob(self, value, **kwargs):
        return self.distribution.prob(value, **kwargs)

    def sample(self, sample_shape=(), **kwargs):
        return self.distribution.sample(sample_shape=sample_shape, **kwargs)

    def stddev(self, **kwargs):
        return self.distribution.stddev(**kwargs)

    def variance(self, **kwargs):
        return self.distribution.variance(**kwargs)
