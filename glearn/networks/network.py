import tensorflow as tf
from glearn.networks.layer import load_layer


class Network(object):
    def __init__(self, name, context, definition, trainable=True):
        self.name = name
        self.context = context
        self.definition = definition
        self.trainable = trainable

        self.layers = {}
        self.head = None
        self.tail = None

    @property
    def seed(self):
        return self.context.seed

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

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def build(self, inputs, outputs=None):
        # all layers within network scope
        with tf.variable_scope(self.name):
            # prepare inputs
            self.tail = inputs

            # create and link network layers
            y = self.tail
            layer_definitions = self.definition.get("layers", [])
            optimizes = self.trainable and self.definition.get("optimizes", True)
            layer_count = len(layer_definitions)
            for i, layer_config in enumerate(layer_definitions):
                layer = load_layer(self, i, layer_config)
                self.add_layer(layer)
                Y = outputs if optimizes and i == layer_count - 1 else None
                y = layer.build(y, Y)
        self.head = y
        return y

    def prepare_default_feeds(self, graphs, feed_map):
        # add default feed values
        for layer_type, type_layers in self.layers.items():
            for layer in type_layers:
                feed_map = layer.prepare_default_feeds(graphs, feed_map)
        return feed_map

    def prob(self, output):
        # HACK - figure better way of exposing this
        distribution = self.get_layer("DistributionLayer")
        return distribution.prob(output)

    def log_prob(self, output):
        # HACK - figure better way of exposing this
        distribution = self.get_layer("DistributionLayer")
        return distribution.log_prob(output)
