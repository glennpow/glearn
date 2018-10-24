from glearn.networks.layers.layer import NetworkLayer


class DistributionLayer(NetworkLayer):
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
