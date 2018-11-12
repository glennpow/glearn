from glearn.policies.policy import Policy
from glearn.networks import load_network


class NetworkPolicy(Policy):
    def __init__(self, config, network, **kwargs):
        self.network_definition = network

        super().__init__(config, **kwargs)

    def build_predict(self):
        # build predict network
        self.network = load_network("policy", self, self.network_definition)
        predict = self.network.build(self.inputs)

        self.set_fetch("predict", predict, ["predict", "evaluate", "debug"])

    def build_loss(self):
        # build loss
        loss, accuracy = self.network.build_loss(self.outputs)

        self.set_fetch("loss", loss, ["evaluate"])
        self.set_fetch("accuracy", accuracy, ["evaluate"])

    def prepare_default_feeds(self, graphs, feed_map):
        # add default feed values
        return self.network.prepare_default_feeds(graphs, feed_map)
