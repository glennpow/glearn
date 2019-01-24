import tensorflow as tf
from tensorflow.python.ops import math_ops
from glearn.utils.log import Loggable
from glearn.networks.context import num_variable_parameters
from glearn.networks.layers.layer import load_layer
from glearn.networks.layers.distributions.distribution import DistributionLayer


class Network(Loggable):
    def __init__(self, name, context, definition, trainable=True):
        self.name = name
        self.context = context
        self.definition = definition
        self.trainable = trainable

        self.layers = []
        self.head = None
        self.tail = None

        self.config = self.context.config
        self.debug_activations = self.config.is_debugging("debug_activations")

    @property
    def seed(self):
        return self.context.seed

    @property
    def summary(self):
        return self.context.summary

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

    def global_variables(self):
        # get all global variables in network scope
        return tf.global_variables(scope=self.name)

    def model_variables(self):
        # get all model variables in network scope
        return tf.model_variables(scope=self.name)

    def trainable_variables(self):
        # get all trainable variables in network scope
        return tf.trainable_variables(scope=self.name)

    def num_global_parameters(self):
        # get total global parameters
        return num_variable_parameters(self.global_variables())

    def num_model_parameters(self):
        # get total model parameters
        return num_variable_parameters(self.model_variables())

    def num_trainable_parameters(self):
        # get total trainable parameters
        return num_variable_parameters(self.trainable_variables())

    def build_predict(self, inputs):
        # all layers within network scope
        with tf.variable_scope(self.name):
            # prepare inputs
            self.tail = inputs

            # create and link network layers
            y = self.tail
            layer_definitions = self.definition.get("layers", [])
            for i, layer_config in enumerate(layer_definitions):
                layer = load_layer(self, i, layer_config)
                layer.inputs = y
                y = layer.build(y)
                layer.outputs = y
                self.add_layer(layer)
            predict = self.get_output_layer().build_predict(y)
            self.head = predict

            # add activation summary for layer
            if self.debug_activations:
                for layer in self.layers:
                    layer.activation_summary(query="evaluate")

        return predict

    def add_loss(self, loss):
        tf.add_to_collection(f"{self.name}_losses", loss)

    def get_total_loss(self):
        # add all losses to get total
        losses = tf.get_collection(f"{self.name}_losses")
        if len(losses) == 0:
            self.warning(f"No loss found for network: '{self.name}'")
            return tf.constant(0, dtype=tf.float32)
        return math_ops.add_n(losses, name="total_loss")

    def build_loss(self, outputs):
        # build prediction loss
        with tf.name_scope("loss"):
            predict_loss, accuracy = self.get_output_layer().build_loss(outputs)
            self.add_loss(predict_loss)

            # build combined total loss
            total_loss = self.get_total_loss()

        self.summary.add_scalar("accuracy", accuracy, "evaluate")

        return total_loss, accuracy

    def prepare_default_feeds(self, queries, feed_map):
        # add default feed values
        for layer in self.layers:
            feed_map = layer.prepare_default_feeds(queries, feed_map)
        return feed_map
