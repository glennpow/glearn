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
