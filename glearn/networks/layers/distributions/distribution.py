import tensorflow as tf
from glearn.networks.layers.layer import NetworkLayer


class DistributionLayer(NetworkLayer):
    def __init__(self, network, index, ent_coef=1e-5):
        super().__init__(network, index)

        self.ent_coef = ent_coef

    def build_predict(self, y):
        # entropy exploration factor
        if self.ent_coef > 0:
            entropy = self.distribution.entropy()
            self.context.set_fetch("entropy", entropy, "evaluate")
            self.context.summary.add_scalar("entropy", tf.reduce_mean(entropy), "evaluate")

            entropy_loss = -self.ent_coef * entropy
            self.add_loss(entropy_loss)
            entropy_loss = tf.reduce_mean(entropy_loss)
            self.context.summary.add_scalar("entropy_loss", entropy_loss, "evaluate")

        return y

    @property
    def distribution(self):
        return self.references["distribution"]

    def covariance(self):
        return self.distribution.covariance()

    def cross_entropy(self, other):
        return self.distribution.cross_entropy(other)

    def entropy(self):
        return self.distribution.entropy()

    def kl_divergence(self, other):
        return self.distribution.kl_divergence(other)

    def log_prob(self, output):
        return self.distribution.log_prob(output)

    def mean(self):
        return self.distribution.mean()

    def mode(self):
        return self.distribution.mode()

    def prob(self, output):
        return self.distribution.prob(output)

    def sample(self, sample_shape=()):
        return self.distribution.sample(sample_shape=sample_shape, seed=self.seed)

    def stddev(self):
        return self.distribution.stddev()

    def variance(self):
        return self.distribution.variance()
