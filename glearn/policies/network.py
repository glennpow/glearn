from glearn.policies.policy import Policy
from glearn.networks import load_network


class NetworkPolicy(Policy):
    def __init__(self, config, network, **kwargs):
        self.network_definition = network

        super().__init__(config, **kwargs)

    def get_info(self):
        info = super().get_info()
        info.update({
            "Global Parameters": self.network.num_global_parameters(),
            "Trainable Parameters": self.network.num_trainable_parameters(),
        })
        return info

    def build_predict(self):
        # build predict network
        self.network = load_network("policy", self, self.network_definition)
        predict = self.network.build_predict(self.inputs)

        self.set_fetch("predict", predict, ["predict", "evaluate"])

    def build_loss(self):
        # build loss
        loss, accuracy = self.network.build_loss(self.outputs)

        self.set_fetch("loss", loss, ["evaluate"])
        self.set_fetch("accuracy", accuracy, ["evaluate"])

    def prepare_default_feeds(self, queries, feed_map):
        feed_map = super().prepare_default_feeds(queries, feed_map)

        # add default feed values
        return self.network.prepare_default_feeds(queries, feed_map)
