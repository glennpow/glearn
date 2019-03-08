import tensorflow as tf
from glearn.policies.policy import Policy
from glearn.networks import load_network


class NetworkPolicy(Policy):
    def __init__(self, config, context, network, **kwargs):
        self.network_definition = network

        super().__init__(config, context, **kwargs)

    def get_info(self):
        info = super().get_info()
        info.update(self.network.get_info())
        return info

    def build_predict(self, inputs):
        # build predict network
        self.network = load_network("policy", self, self.network_definition)
        predict = self.network.build_predict(inputs)

        # clip output
        if self.config.output.continuous:
            with tf.name_scope(f"{self.network.scope}/"):
                output_space = self.config.output.space
                predict = tf.clip_by_value(predict, output_space.low, output_space.high)
                self.network.outputs = predict

        self.inputs = inputs
        self.add_fetch("predict", predict, ["predict", "evaluate"])

    def build_loss(self, outputs):
        # build loss
        loss, accuracy = self.network.build_loss(outputs)

        self.outputs = outputs
        self.add_fetch("loss", loss, ["evaluate"])
        self.add_fetch("accuracy", accuracy, ["evaluate"])

    def optimize_loss(self, loss=None):
        return self.network.optimize_loss(loss)

    def prepare_default_feeds(self, queries, feed_map):
        feed_map = super().prepare_default_feeds(queries, feed_map)

        # add default feed values
        return self.network.prepare_default_feeds(queries, feed_map)
