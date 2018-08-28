import numpy as np
import tensorflow as tf
import pyglet
from policies.policy import Policy
from policies.layers import add_fc, add_conv2d


class CNN(Policy):
    def __init__(self, learning_rate=2e-4, filters=[(5, 5, 32), (5, 5, 64)], strides=1,
                 padding="SAME", max_pool_k=2, fc_layers=[7 * 7 * 64, 1024], keep_prob=0.8,
                 **kwargs):
        self.learning_rate = learning_rate  # lamdba λ
        # self.discount_factor = discount_factor  # gamma γ

        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.max_pool_k = max_pool_k

        self.fc_layers = fc_layers
        self.keep_prob = keep_prob

        self.visualize_layer = None
        self.visualize_feature = None

        super().__init__(**kwargs)

        self.init_visualize()

    def init_model(self):
        # create input placeholders
        with tf.name_scope('inputs'):
            inputs, outputs = self.create_inputs()
            self.dropout = tf.placeholder(tf.float32, (), name="dropout")
            # self.discounted_rewards = tf.placeholder(tf.float32, (None, ), name="V")

        # prepare inputs
        # inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])  # TODO - pass/infer dimensions arg?
        layer = tf.cast(inputs, tf.float32)
        input_size = self.input.size  # FIXME - can we always infer this from inputs?
        input_channels = self.input.shape[2]

        # create conv layers
        for i, filter in enumerate(self.filters):
            layer, info = add_conv2d(self, layer, input_size, input_channels, filter,
                                     strides=self.strides, max_pool_k=self.max_pool_k,
                                     padding=self.padding)
            input_size = layer.shape[1]
            input_channels = filter[2]

        # create fully connected layers
        input_size = np.prod(layer.shape[1:])
        for i, fc_size in enumerate(self.fc_layers):
            layer, info = add_fc(self, layer, input_size, fc_size, reshape=i == 0,
                                 keep_prob=self.dropout)
            input_size = layer.shape[1]

        # create output layer
        layer, info = add_fc(self, layer, input_size, self.output.size, activation=tf.nn.softmax)

        # store action
        act = layer
        self.act_graph["act"] = act

        # calculate loss
        with tf.name_scope('loss'):
            logits = info["Z"]
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
            # loss = tf.reduce_mean(neg_log_p * self.discounted_rewards)  # TODO?
            loss = tf.reduce_mean(neg_log_p)
            self.evaluate_graph["loss"] = loss

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize_graph["optimize"] = optimizer.minimize(loss)

        # evaluate accuracy
        with tf.name_scope('predict'):
            self.evaluate_graph["act"] = act
            if self.output.discrete:
                # TODO - stochastic discrete also
                evaluate = tf.equal(tf.argmax(act, 1), tf.argmax(outputs, 1))
                accuracy = tf.reduce_mean(tf.cast(evaluate, tf.float32))
                self.evaluate_graph["accuracy"] = accuracy
            else:
                # TODO - continuous evaluate
                pass

    def act_feed(self, observation, feed_dict):
        # no dropout for inference
        feed_dict[self.dropout] = 1

        return feed_dict

    def optimize_feed(self, data, feed_dict):
        feed_dict[self.dropout] = self.keep_prob

        return feed_dict

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
            self.act_graph[f"conv2d_{i}"] = self.get_layer("conv2d", i)

    def update_visualize(self, data):
        index = 0
        image = data.inputs[index] * 255
        self.set_main_image(image)

        action = self.output.decode(self.evaluate_result["act"][index])
        action_message = f"{action}"
        self.add_label("action", action_message)

    def clear_visualize(self):
        self.visualize_layer = None
        self.remove_image("features")

    def rollout(self):
        # do standard rollout
        transition = super().rollout()

        if self.visualize_layer is not None:
            # get layer values to visualize
            values = self.act_result[f"conv2d_{self.visualize_layer}"]

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

    def optimize(self, evaluating=False, saving=True):
        data = super().optimize(evaluating=evaluating, saving=saving)

        # visualize evaluated dataset results
        if self.supervised and evaluating:
            self.update_visualize(data)
