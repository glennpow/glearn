import numpy as np
import tensorflow as tf
import pyglet
from glearn.policies.policy import Policy


class CNNPolicy(Policy):
    def __init__(self, config, version=None):
        self.filters = config.get("filters", [(5, 5, 32), (5, 5, 64)])
        self.strides = config.get("strides", 1)
        self.padding = config.get("padding", "SAME")
        self.max_pool_k = config.get("max_pool_k", 2)

        self.fc_layers = config.get("fc_layers", [7 * 7 * 64, 1024])
        self.keep_prob = config.get("keep_prob", 0.8)

        self.visualize_grid = config.get("visualize_grid", [1, 1])
        self.visualize_graph = None
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

    def predict(self, data):
        result = super().predict(data)

        self.visualize_features("predict")

        return result

    def optimize(self, step):
        data, results = super().optimize(step)

        # visualize evaluated dataset results
        if self.supervised and self.evaluating:
            self.visualize_features("evaluate")

            self.update_visualize(data)

    def handle_key_press(self, key, modifiers):
        super().handle_key_press(key, modifiers)

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
            self.visualize_features()
        elif key == pyglet.window.key.MINUS:
            if self.visualize_layer is not None:
                self.visualize_layer -= 1
                if self.visualize_layer < 0:
                    self.clear_visualize()
                else:
                    max_features = self.filters[self.visualize_layer][2]
                    self.visualize_feature = min(self.visualize_feature, max_features - 1)
                    self.visualize_features()
        elif key == pyglet.window.key.BRACKETRIGHT:
            if self.visualize_layer is not None:
                max_features = self.filters[self.visualize_layer][2]
                self.visualize_feature = min(self.visualize_feature + 1, max_features - 1)
                self.visualize_features()
        elif key == pyglet.window.key.BRACKETLEFT:
            if self.visualize_layer is not None:
                self.visualize_feature = max(self.visualize_feature - 1, 0)
                self.visualize_features()

    def init_visualize(self):
        if self.viewer is not None:
            for i in range(len(self.filters)):
                self.set_fetch(f"conv2d_{i}", self.get_layer("conv2d", i), ["predict", "evaluate"])

    def update_visualize(self, data):
        # build image grid of inputs
        grid = self.visualize_grid + [1]
        image_size = np.multiply(self.input.shape, grid)
        width = self.input.shape[1]
        height = self.input.shape[0]
        image = np.zeros(image_size)
        for row in range(grid[0]):
            for col in range(grid[1]):
                index = row * grid[1] + col
                input_image = data.inputs[index] * 255
                x = col * width
                y = row * height
                image[y:y + height, x:x + width] = input_image

                action = self.output.decode(self.results["evaluate"]["predict"][index])
                action_s = f"{action}"
                correct = action == self.output.decode(data.outputs[index])
                color = (0, 255, 0, 255) if correct else (255, 0, 0, 255)
                lx = x + width
                ly = image_size[0] - (y + height)
                self.add_label(f"action:{index}", action_s, x=lx, y=ly, font_size=8, color=color,
                               anchor_x='right', anchor_y='bottom')
        self.set_main_image(image)

    def visualize_features(self, graph=None):
        if graph is None:
            graph = self.visualize_graph
        self.visualize_graph = graph
        if self.viewer is not None and self.visualize_layer is not None:
            if graph in self.results:
                # get layer values to visualize
                values = self.results[graph][f"conv2d_{self.visualize_layer}"]
                flat_values = values.ravel()
                value_min = min(flat_values)
                value_max = max(flat_values)
                value_range = max([0.1, value_max - value_min])

                # build grid of feature images
                vh = self.viewer.height
                width = self.input.shape[1]
                height = self.input.shape[0]
                for row in range(self.visualize_grid[0]):
                    for col in range(self.visualize_grid[1]):
                        # build image for selected feature
                        index = row * self.visualize_grid[1] + col
                        _, f_height, f_width, _ = values.shape
                        image = np.zeros((f_height, f_width, 1))
                        for y, f_row in enumerate(values[index]):
                            for x, f_col in enumerate(f_row):
                                value = f_col[self.visualize_feature]
                                image[y][x][0] = int((value - value_min) / value_range * 255)

                        # add image
                        x = col * width
                        y = vh - (row + 1) * height
                        self.add_image(f"feature:{index}", image, x=x, y=y, width=width,
                                       height=height)

    def clear_visualize(self):
        self.visualize_layer = None
        self.remove_images("feature")
