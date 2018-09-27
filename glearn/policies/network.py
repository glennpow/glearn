import tensorflow as tf
from glearn.policies.policy import Policy
from glearn.network import load_layer


class NetworkPolicy(Policy):
    def __init__(self, config, layers=[], **kwargs):
        self.layer_configs = layers

        self.layers = {}

        super().__init__(config, **kwargs)

    def add_layer(self, layer):
        layer_type = layer.layer_type
        if layer_type in self.layers:
            type_layers = self.layers[layer_type]
        else:
            type_layers = []
            self.layers[layer_type] = type_layers
        type_layers.append(layer)

    def get_layer(self, layer_type, index=0):
        if layer_type in self.layers:
            type_layers = self.layers[layer_type]
            if index < len(type_layers):
                return type_layers[index]
        return None

    def get_layer_count(self, layer_type=None):
        if layer_type is None:
            return len(self.layers)
        else:
            if layer_type in self.layers:
                return len(self.layers[layer_type])
        return 0

    def init_model(self):
        # create input placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

            self.set_fetch("X", inputs, ["evaluate", "debug"])
            self.set_fetch("Y", outputs, "evaluate")

        # create and link network layers
        x = inputs
        layer_count = len(self.layer_configs)
        for i, layer_config in enumerate(self.layer_configs):
            layer = load_layer(i, layer_config)
            self.add_layer(layer)
            Y = outputs if i == layer_count - 1 else None
            x = layer.build(self, x, Y)

        # store prediction
        self.set_fetch("predict", x, ["predict", "evaluate"])

    def prepare_default_feeds(self, graphs, feed_map):
        # add default feed values
        for layer_type, type_layers in self.layers.items():
            for layer in type_layers:
                feed_map = layer.prepare_default_feeds(graphs, feed_map)
        return feed_map
