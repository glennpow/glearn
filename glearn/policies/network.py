import tensorflow as tf
from glearn.policies.policy import Policy


class NetworkPolicy(Policy):
    def __init__(self, config, context, network, scale_output=False, **kwargs):
        self.network_definition = network
        self.scale_output = scale_output  # HACK - figure out way to avoid needing this

        super().__init__(config, context, **kwargs)

    def get_info(self):
        info = super().get_info()
        info.update(self.network.get_info())
        return info

    def build_predict(self, inputs, query=None):
        # build predict network
        self.network = self.context.build_network(self.name, self.network_definition, inputs,
                                                  query=query)

        self.handle_predict(inputs, self.network.outputs)

    def handle_predict(self, inputs, predict):
        with self.variable_scope(self.network.scope):
            if self.config.output.continuous:
                # scale and clip predict
                output_space = self.config.output.space
                low = output_space.low
                high = output_space.high
                if self.scale_output:
                    # this only works with sigmoid activation in last layer
                    predict = predict * (high - low) + low
                predict = tf.clip_by_value(predict, low, high)
                self.network.outputs = predict

            # add fetch for predict
            self.add_metric("predict", predict, histogram=True, query=["predict", "evaluate"])

        # remember values
        self.inputs = inputs
        self.predict = predict

    def build_loss(self, targets):
        # build policy network loss
        self.targets = targets
        return self.network.build_loss(targets)

    def optimize_loss(self, loss, name=None):
        return self.network.optimize_loss(loss, name=name)

    def optimize_error(self, target, predict=None, loss_type=None, weights=None, name=None):
        return self.network.optimize_error(target, predict=predict, loss_type=loss_type,
                                           weights=weights, name=name)

    def prepare_default_feeds(self, query, feed_map):
        feed_map = super().prepare_default_feeds(query, feed_map)

        # add default feed values
        return self.network.prepare_default_feeds(query, feed_map)
