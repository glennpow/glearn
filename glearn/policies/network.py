from glearn.policies.policy import Policy
from glearn.networks import load_network


class NetworkPolicy(Policy):
    def __init__(self, config, network, **kwargs):
        self.network_definition = network

        super().__init__(config, **kwargs)

    def build_predict(self, inputs, outputs):
        # build network
        self.network = load_network("policy", self, self.network_definition)
        return self.network.build(inputs, outputs)

    def prepare_default_feeds(self, graphs, feed_map):
        # add default feed values
        return self.network.prepare_default_feeds(graphs, feed_map)
