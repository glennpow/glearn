import numpy as np
import tensorflow as tf
import pyglet
from glearn.policies.policy import Policy


class CNN(Policy):
    def __init__(self, config, version=None):
        self.filters = config.get("filters", [(5, 5, 32), (5, 5, 64)])
        self.strides = config.get("strides", 1)
        self.padding = config.get("padding", "SAME")
        self.max_pool_k = config.get("max_pool_k", 2)

        self.fc_layers = config.get("fc_layers", [7 * 7 * 64, 1024])
        self.keep_prob = config.get("keep_prob", 0.8)

        self.visualize_layer = None
        self.visualize_feature = None

        super().__init__(config, version=version)

        self.init_visualize()

    def init_model(self):
        # create input placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

            dropout = tf.placeholder(tf.float32, (), name="dropout")
            self.set_feed("dropout", dropout)

            # gamma = tf.placeholder(tf.float32, (None, ), name="gamma")
            # self.set_feed("gamma", gamma, ["optimize", "evaluate"])

        # prepare inputs
        # inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])  # TODO - pass/infer dimensions arg?
        layer = tf.cast(inputs, tf.float32)
        input_size = self.input.size  # FIXME - can we always infer this from inputs?
        input_channels = self.input.shape[2]

        # create conv layers
        for i, filter in enumerate(self.filters):
            layer, info = self.add_conv2d(layer, input_size, input_channels, filter,
                                          strides=self.strides, max_pool_k=self.max_pool_k,
                                          padding=self.padding)
            input_size = layer.shape[1]
            input_channels = filter[2]

        # create fully connected layers
        input_size = np.prod(layer.shape[1:])
        for i, fc_size in enumerate(self.fc_layers):
            layer, info = self.add_fc(layer, input_size, fc_size, reshape=i == 0,
                                      keep_prob=dropout)
            input_size = layer.shape[1]

        # create output layer
        layer, info = self.add_fc(layer, input_size, self.output.size,
                                  activation=tf.nn.softmax)

        # store prediction
        predict = layer
        self.set_fetch("predict", predict, ["predict", "evaluate"])

        # calculate loss
        with tf.name_scope('loss'):
            logits = info["Z"]
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
            # loss = tf.reduce_mean(neg_log_p * self.discounted_rewards)  # TODO?
            loss = tf.reduce_mean(neg_log_p)
            self.set_fetch("loss", loss, "evaluate")

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.set_fetch("optimize", optimizer.minimize(loss), "optimize")

        # evaluate accuracy
        with tf.name_scope('accuracy'):
            if self.output.discrete:
                # TODO - stochastic discrete also
                evaluate = tf.equal(tf.argmax(predict, 1), tf.argmax(outputs, 1))
                accuracy = tf.reduce_mean(tf.cast(evaluate, tf.float32))
                self.set_fetch("accuracy", accuracy, "evaluate")
            else:
                # TODO - continuous evaluate
                pass

    def prepare_feed_map(self, graph, data, feed_map):
        if graph == "optimize":
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1
        return feed_map

    def rollout(self):
        # do standard rollout
        transition = super().rollout()

        if self.visualize_layer is not None:
            # get layer values to visualize
            values = self.results["predict"][f"conv2d_{self.visualize_layer}"]

            # build image for selected feature
            _, rows, cols, _ = values.shape
            image = np.zeros((rows, cols, 1))
            flat_values = values.ravel()
            value_min = min(flat_values)
            value_max = max(flat_values)
            value_range = max([0.1, value_max - value_min])
            for y, row in enumerate(values[0]):
                for x, col in enumerate(row):
                    value = col[self.visualize_feature]
                    image[y][x][0] = int((value - value_min) / value_range * 255)

            # render image
            width, height = self.get_viewer_size()
            self.add_image("features", image, x=0, y=0, width=width, height=height)

        return transition

    def optimize(self, step):
        data, results = super().optimize(step)

        # visualize evaluated dataset results
        if self.supervised and self.evaluating:
            self.update_visualize(data)

    def on_key_press(self, key, modifiers):
        super().on_key_press(key, modifiers)

        # feature visualization keys
        if key == pyglet.window.key._0:
            self.clear_visualize()
        elif key == pyglet.window.key.EQUAL:
            if self.visualize_layer is None:
                self.visualize_layer = 0
                self.visualize_feature = 0
            else:
                max_layers = len(self.filters)
                self.visualize_layer = min(self.visualize_layer + 1, max_layers - 1)
            max_features = self.filters[self.visualize_layer][2]
            self.visualize_feature = min(self.visualize_feature, max_features - 1)
        elif key == pyglet.window.key.MINUS:
            if self.visualize_layer is not None:
                self.visualize_layer -= 1
                if self.visualize_layer < 0:
                    self.clear_visualize()
                else:
                    max_features = self.filters[self.visualize_layer][2]
                    self.visualize_feature = min(self.visualize_feature, max_features - 1)
        elif key == pyglet.window.key.BRACKETRIGHT:
            if self.visualize_layer is not None:
                max_features = self.filters[self.visualize_layer][2]
                self.visualize_feature = min(self.visualize_feature + 1, max_features - 1)
        elif key == pyglet.window.key.BRACKETLEFT:
            if self.visualize_layer is not None:
                self.visualize_feature = max(self.visualize_feature - 1, 0)

    def init_visualize(self):
        for i in range(len(self.filters)):
            self.set_fetch(f"conv2d_{i}", self.get_layer("conv2d", i), "predict")

    def update_visualize(self, data):
        index = 0
        image = data.inputs[index] * 255
        self.set_main_image(image)

        action = self.output.decode(self.results["evaluate"]["predict"][index])
        action_message = f"{action}"
        self.add_label("action", action_message)

    def clear_visualize(self):
        self.visualize_layer = None
        self.remove_image("features")