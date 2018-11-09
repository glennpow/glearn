import tensorflow as tf
from glearn.networks.layers.layer import load_layer
from glearn.networks.losses import load_loss
from glearn.networks.layers.distributions.distribution import DistributionLayer


class Network(object):
    def __init__(self, name, context, definition, trainable=True):
        self.name = name
        self.context = context
        self.definition = definition
        self.trainable = trainable

        self.layers = []
        self.head = None
        self.tail = None

    @property
    def seed(self):
        return self.context.seed

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_layer(self, layer_type=None, index=0):
        i = 0
        for layer in self.layers:
            if layer_type is None or isinstance(layer, layer_type):
                if i == index:
                    return layer
                i += 1
        return None

    def get_output_layer(self):
        return self.layers[-1]

    def get_layers(self, layer_type=None):
        if layer_type is None:
            return self.layers
        layers = []
        for layer in self.layers:
            if isinstance(layer, layer_type):
                layers.append(layer)
        return layers

    def get_distribution_layer(self):
        # look for distribution layer
        return self.get_layer(DistributionLayer)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def build(self, inputs):
        # all layers within network scope
        with tf.variable_scope(self.name):
            # prepare inputs
            self.tail = inputs

            # create and link network layers
            y = self.tail
            layer_definitions = self.definition.get("layers", [])
            for i, layer_config in enumerate(layer_definitions):
                layer = load_layer(self, i, layer_config)
                self.add_layer(layer)
                y = layer.build(y)
            predict = self.get_output_layer().build_predict(y)
        self.head = predict
        return predict

    def build_loss(self, outputs, **kwargs):
        # build loss
        loss_definition = self.definition.get("loss", None)
        with tf.name_scope(self.name):
            return load_loss(loss_definition, self, outputs, **kwargs)

    def prepare_default_feeds(self, graphs, feed_map):
        # add default feed values
        for layer in self.layers:
            feed_map = layer.prepare_default_feeds(graphs, feed_map)
        return feed_map
