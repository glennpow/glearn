import tensorflow as tf
from glearn.policies.policy import Policy
from glearn.networks import load_network


class NetworkPolicy(Policy):
    def __init__(self, config, context, network, scale_output=False, **kwargs):
        self.network_definition = network
        self.scale_output = scale_output  # HACK - figure out way to avoid this

        super().__init__(config, context, **kwargs)

    def get_info(self):
        info = super().get_info()
        info.update(self.network.get_info())
        return info

    def build_predict(self, inputs):
        # build predict network
        self.network = load_network("policy", self.context, self.network_definition)
        predict = self.network.build_predict(inputs)

        # scale and clip output
        if self.config.output.continuous:
            with tf.name_scope(f"{self.network.scope}/"):
                output_space = self.config.output.space
                low = output_space.low
                high = output_space.high
                if self.scale_output:
                    # this only works with sigmoid activation in last layer
                    predict = predict * (high - low) + low
                predict = tf.clip_by_value(predict, low, high)

                self.network.outputs = predict

        self.inputs = inputs
        self.outputs = predict
        self.add_fetch("predict", predict, ["predict", "evaluate"])

    def build_loss(self, targets):
        # build policy network loss
        self.targets = targets
        return self.network.build_loss(targets)

    def optimize_loss(self, loss, name=None):
        return self.network.optimize_loss(loss, name=name)

    def prepare_default_feeds(self, query, feed_map):
        feed_map = super().prepare_default_feeds(query, feed_map)

        # add default feed values
        return self.network.prepare_default_feeds(query, feed_map)
