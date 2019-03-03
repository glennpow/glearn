import tensorflow as tf
from tensorflow.python.ops import math_ops
from glearn.utils.config import Configurable
from glearn.networks.context import num_variable_parameters
from glearn.networks.layers.layer import load_layer
from glearn.networks.layers.distributions.distribution import DistributionLayer


class Network(Configurable):
    def __init__(self, name, context, definition, trainable=True):
        self.name = name
        self.scope = name
        self.context = context
        self.definition = definition
        self.trainable = trainable

        self.layers = []
        self.inputs = None
        self.outputs = None
        self.loss = None
        self.accuracy = None

        super().__init__(self.context.config)

        self.debug_activations = self.config.is_debugging("debug_activations")
        self.debug_gradients = self.config.is_debugging("debug_gradients")

    def get_info(self):
        return {
            "Global Parameters": self.num_global_parameters(),
            "Trainable Parameters": self.num_trainable_parameters(),
        }

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
        return tf.global_variables(scope=self.scope)

    def model_variables(self):
        # get all model variables in network scope
        return tf.model_variables(scope=self.scope)

    def trainable_variables(self):
        # get all trainable variables in network scope
        return tf.trainable_variables(scope=self.scope)

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

    def build_predict(self, inputs):
        # all layers within network scope
        with tf.variable_scope(self.get_scope_name()):
            self.scope = tf.get_variable_scope().name

            # create and link network layers
            y = inputs
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

    def build_loss(self, outputs):
        # build prediction loss
        with tf.name_scope("loss"):
            predict_loss, self.accuracy = self.get_output_layer().build_loss(outputs)
            self.add_loss(predict_loss)

            # build combined total loss
            self.loss = self.build_total_loss()

        self.summary.add_scalar("total_loss", self.loss)
        self.summary.add_scalar("accuracy", self.accuracy)

        return self.loss, self.accuracy

    def optimize_loss(self, loss=None):
        # prepare loss
        if loss is None:
            loss = self.loss
        else:
            self.loss = loss
        if loss is None:
            self.error(f"Network '{self.name}' doesn't define a loss to optimize")
            return None

        learning_rate = self.definition.get("learning_rate", 1e-4)

        # learning rate decay
        lr_decay = self.definition.get("lr_decay", None)
        if lr_decay is not None:
            lr_decay_intervals = self.definition.get("lr_decay_intervals", 1)
            decay_steps = int(lr_decay_intervals * self.config.get_interval_size())
            learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                       decay_steps, lr_decay, staircase=True)

        # create optimizer
        optimizer_name = self.definition.get("optimizer", "sgd")
        if optimizer_name == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception(f"Unknown optimizer type specified in config: {optimizer_name}")

        # get gradients and trainable variables
        grads_tvars = optimizer.compute_gradients(loss)
        grads_tvars = [(g, v) for (g, v) in grads_tvars if g is not None]

        # validate affected network variables during optimization
        if self.debug:
            def report_warnings(title, variables):
                if len(variables) > 0:
                    self.warning(f"\n{title} variables during network optimization: {self.name}")
                    for v in variables:
                        self.warning(f" * {v.name}  |  {v.shape}")

            expected_vars = self.trainable_variables()
            computed_vars = [v for (g, v) in grads_tvars]
            missing = [v for v in computed_vars if v not in computed_vars]
            unexpected = [v for v in computed_vars if v not in expected_vars]
            report_warnings("Missing", missing)
            report_warnings("Unexpected", unexpected)

        # check if we require unzipping grad/vars
        max_grad_norm = self.definition.get("max_grad_norm", None)
        require_unzip = self.debug_gradients or max_grad_norm is not None
        if require_unzip:
            grads, tvars = zip(*grads_tvars)

        # apply gradient clipping
        if max_grad_norm is not None:
            with tf.name_scope("clipped_gradients"):
                grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)

                # metrics to observe clipped gradient ratio and global norm
                if self.debug_gradients:
                    clipped_ratio = tf.maximum(global_norm - self.clip_norm, 0) / global_norm
                    self.summary.add_scalar("global_norm", global_norm)
                    self.summary.add_scalar("clipped_ratio", clipped_ratio)

        if require_unzip:
            grads_tvars = zip(grads, tvars)

        # apply gradients
        optimize = optimizer.apply_gradients(grads_tvars)

        # add learning rate and gradient summaries
        self.summary.add_scalar("learning_rate", learning_rate)
        if self.debug_gradients:
            self.summary.add_gradients(grads_tvars)

        return optimize

    def prepare_default_feeds(self, queries, feed_map):
        # add default feed values
        for layer in self.layers:
            feed_map = layer.prepare_default_feeds(queries, feed_map)
        return feed_map

    def clone(self, name, inputs=None, trainable=False):
        # build duplicate network
        from glearn.networks import load_network
        network = load_network(name, self.context, self.definition, trainable=trainable)

        if inputs is None:
            inputs = self.inputs
        network.build_predict(inputs)
        return network

    def update(self, network, tau=0, name=None):
        if name is None:
            name = f"{self.name}_update"
        with tf.name_scope(name):
            # build target policy update
            source_vars = network.global_variables()
            target_vars = self.global_variables()
            if tau <= 0:
                updates = [tp.assign(p) for p, tp in zip(source_vars, target_vars)]
            else:
                updates = [tp.assign(tp * (1.0 - tau) + p * tau)
                           for p, tp in zip(source_vars, target_vars)]
            network_update = tf.group(*updates, name="update_parameters")
            self.context.add_fetch(name, network_update)
            return network_update
