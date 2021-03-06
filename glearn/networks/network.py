import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from glearn.utils.config import Configurable
from glearn.utils import tf_utils
from glearn.networks.context import num_variable_parameters
from glearn.networks.layers.layer import load_layer
from glearn.networks.layers.distributions.distribution import DistributionLayer


class Network(Configurable):
    def __init__(self, name, context, definition, trainable=True):
        self.name = name
        self.context = context
        self.definition = definition
        self.trainable = trainable

        self.scope = None
        self.layers = []
        self.inputs = None
        self.outputs = None

        super().__init__(self.context.config)

        self.debug_activations = self.is_debugging("debug_activations")

    def get_info(self):
        return {
            "Global Parameters": self.num_global_parameters(),
            "Trainable Parameters": self.num_trainable_parameters(),
        }

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_layer(self, index):
        return self.layers[index]

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

    def find_layer(self, layer_type, index=0):
        i = 0
        for layer in self.layers:
            if layer_type is None or isinstance(layer, layer_type):
                if i == index:
                    return layer
                i += 1
        return None

    def get_distribution_layer(self):
        # look for distribution layer
        return self.find_layer(DistributionLayer)

    def global_variables(self):
        # get all global variables in network scope
        return tf.global_variables(scope=self.scope.name)

    def model_variables(self):
        # get all model variables in network scope
        return tf.model_variables(scope=self.scope.name)

    def trainable_variables(self):
        # get all trainable variables in network scope
        return tf.trainable_variables(scope=self.scope.name)

    def num_global_parameters(self):
        # get total global parameters
        return num_variable_parameters(self.global_variables())

    def num_model_parameters(self):
        # get total model parameters
        return num_variable_parameters(self.model_variables())

    def num_trainable_parameters(self):
        # get total trainable parameters
        return num_variable_parameters(self.trainable_variables())

    def get_scope_name(self):
        return f"{self.name}_network"

    def variable_scope(self, name_or_scope, **kwargs):
        return self.context.variable_scope(name_or_scope, **kwargs)

    def build_predict(self, inputs, reuse=None):
        # all layers within network scope
        with self.variable_scope(self.get_scope_name(), reuse=reuse) as scope:
            self.scope = scope

            # prepare inputs
            y = inputs

            # create and link network layers
            layer_definitions = self.definition.get("layers", [])
            for i, layer_config in enumerate(layer_definitions):
                layer = load_layer(self, i, layer_config)
                layer.inputs = y
                y = layer.build(y)
                layer.outputs = y
                self.add_layer(layer)
            predict = self.get_output_layer().build_predict(y)

            self.inputs = inputs
            self.outputs = predict

            # add activation summary for layer
            if self.debug_activations:
                for layer in self.layers:
                    layer.activation_summary(query="evaluate")

        return predict

    def add_loss(self, loss):
        tf.add_to_collection(f"{self.name}_losses", loss)

    def add_regularization_loss(self, factor):
        # TODO - work on this a bit...
        loss = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.trainable_variables()]) * factor
        self.add_loss(loss)

    def build_total_loss(self):
        # add all losses to get total
        losses = tf.get_collection(f"{self.name}_losses")
        if len(losses) == 0:
            self.warning(f"No loss found for network: '{self.name}'")
            return tf.constant(0, dtype=tf.float32)
        return math_ops.add_n(losses, name="total_loss")

    def build_loss(self, targets):
        # build prediction loss
        predict_metrics = {}
        with self.variable_scope(f"{self.name}_loss"):
            predict_loss, metrics = self.get_output_layer().build_loss(targets)
            self.add_loss(predict_loss)
            predict_metrics.update(metrics)

            # build combined total loss
            loss = self.build_total_loss()

        self.context.add_metric("loss", loss)
        for k, v in predict_metrics.items():
            self.context.add_metric(k, v)

        return loss

    def optimize_loss(self, loss, name=None):
        return self.context.optimize_loss(loss, networks=[self], name=name)

    def optimize_error(self, target, predict=None, loss_type=None, weights=None, name=None):
        if name is None:
            name = f"{self.name}_optimize"
        with self.variable_scope(name):
            with self.variable_scope("loss"):
                if predict is None:
                    # default to network outputs
                    predict = self.outputs

                # mean squared error loss
                error = predict - tf.stop_gradient(target)

                # calculate appropriate loss
                if loss_type == "huber":
                    error = tf_utils.huber_loss(error)
                else:  # default "mean_square_error"
                    error = tf.square(error)

                # allow weighting
                if weights is not None:
                    error *= weights

                loss = tf.reduce_mean(error)

            # summaries
            self.context.add_metric(f"{self.name}_target", target, histogram=True, query=name)
            self.context.add_metric(f"{self.name}_loss", loss, query=name)

            # minimize V-loss
            return self.optimize_loss(loss, name=name), loss

    def prepare_default_feeds(self, query, feed_map):
        # add default feed values
        for layer in self.layers:
            feed_map = layer.prepare_default_feeds(query, feed_map)
        return feed_map

    def clone(self, name, inputs=None, trainable=False):
        # build duplicate network
        from glearn.networks import load_network
        network = load_network(name, self.context, self.definition, trainable=trainable)

        if inputs is None:
            inputs = self.inputs
        network.build_predict(inputs)
        return network

    def update(self, network, tau=None, name=None, query=None):
        if name is None:
            name = f"{self.name}_update"
        with self.variable_scope(name):
            # build target network update
            source_vars = network.global_variables()
            target_vars = self.global_variables()
            var_map = zip(source_vars, target_vars)
            if tau is None:
                updates = [tp.assign(p) for p, tp in var_map]
            else:
                tau = np.clip(tau, 0, 1)
                updates = [tp.assign(tp * (1.0 - tau) + p * tau) for p, tp in var_map]
            network_update = tf.group(*updates, name="update_parameters")
            self.context.add_fetch(name, network_update, query=query)
            return network_update
