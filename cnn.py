import numpy as np
import tensorflow as tf
import pyglet
from policy import Policy
from layers import add_fc, add_conv2d


class CNN(Policy):
    # filters: [height, width, chan_in, chan_out]
    def __init__(self, env=None, dataset=None, batch_size=128, seed=0, learning_rate=2e-4,
                 filters=[(5, 5, 32), (5, 5, 64)], strides=1, padding="SAME", max_pool_k=2,
                 fc_layers=[7 * 7 * 64, 1024], dropout_rate=0.8, load_path=None, save_path=None,
                 tensorboard_path=None):
        self.learning_rate = learning_rate  # lamdba λ
        # self.discount_factor = discount_factor  # gamma γ

        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.max_pool_k = max_pool_k

        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate

        super().__init__(env=env, dataset=dataset, batch_size=batch_size, seed=seed,
                         load_path=load_path, save_path=save_path,
                         tensorboard_path=tensorboard_path)

    def init_model(self):
        # create input placeholders
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, (None,) + self.input.shape, name="X")
            self.outputs = tf.placeholder(tf.float32, (None,) + self.output.shape, name="Y")
            self.dropout = tf.placeholder(tf.float32, (), name="dropout")
            # self.discounted_rewards = tf.placeholder(tf.float32, (None, ), name="V")

        # prepare inputs
        inputs = self.inputs
        # inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])  # TODO - pass/infer dimensions arg?
        layer = (inputs, None)
        input_size = self.input.size  # FIXME - can we always infer this from inputs?
        input_channels = self.input.shape[2]

        # create conv layers
        for i, filter in enumerate(self.filters):
            layer = add_conv2d(self, layer[0], input_size, input_channels, filter,
                               strides=self.strides, max_pool_k=self.max_pool_k,
                               padding=self.padding)
            input_size = layer[0].shape[1]
            input_channels = filter[2]

        # create fully connected layers
        input_size = np.prod(layer[0].shape[1:])
        for i, fc_size in enumerate(self.fc_layers):
            layer = add_fc(self, layer[0], input_size, fc_size, reshape=i == 0,
                           dropout=self.dropout)
            input_size = layer[0].shape[1]

        # create output layer
        layer = add_fc(self, layer[0], input_size, self.output.size,
                       activation=tf.nn.softmax)

        # softmax
        logits = layer[1]["Z"]
        labels = self.outputs
        act = tf.nn.softmax(logits, name='act')
        self.act_graph["act"] = act

        self.init_visualize()

        # calculate loss
        with tf.name_scope('loss'):
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            # loss = tf.reduce_mean(neg_log_p * self.discounted_rewards)  # TODO?
            loss = tf.reduce_mean(neg_log_p)
            self.evaluate_graph["loss"] = loss

        # minimize loss
        with tf.name_scope('optimize'):
            optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            self.optimize_graph["optimize"] = optimize

        # evaluate accuracy
        with tf.name_scope('evaluate'):
            if self.output.discrete:
                # TODO - stochastic discrete also
                evaluate = tf.equal(tf.argmax(act, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(evaluate, tf.float32))
                self.evaluate_graph["accuracy"] = accuracy
            else:
                # TODO - continuous evaluate
                pass

    def act_feed(self, observation, feed_dict):
        # no dropout for inference
        feed_dict[self.dropout] = 1

        return feed_dict

    def optimize_feed(self, batch, feed_dict):
        feed_dict[self.dropout] = self.dropout_rate

        return feed_dict

    # HACKs - to visualize embeddings
    # ---------------------------------------------------------------------
    visualize_layer = None
    visualize_embedding = None

    def init_viewer(self):
        super().init_viewer()

        if self.viewer is not None:
            self.viewer.window.push_handlers(self)

    def on_key_press(self, key, modifiers):
        if key == pyglet.window.key._0:
            self.clear_visualize()
        elif key == pyglet.window.key.EQUAL:
            if self.visualize_layer is None:
                self.visualize_layer = 0
                self.visualize_embedding = 0
            else:
                max_layers = len(self.filters)
                self.visualize_layer = min(self.visualize_layer + 1, max_layers - 1)
            max_embeddings = self.filters[self.visualize_layer][2]
            self.visualize_embedding = min(self.visualize_embedding, max_embeddings - 1)
        elif key == pyglet.window.key.MINUS:
            if self.visualize_layer is not None:
                self.visualize_layer -= 1
                if self.visualize_layer < 0:
                    self.clear_visualize()
                else:
                    max_embeddings = self.filters[self.visualize_layer][2]
                    self.visualize_embedding = min(self.visualize_embedding, max_embeddings - 1)
        elif key == pyglet.window.key.BRACKETRIGHT:
            if self.visualize_layer is not None:
                max_embeddings = self.filters[self.visualize_layer][2]
                self.visualize_embedding = min(self.visualize_embedding + 1, max_embeddings - 1)
        elif key == pyglet.window.key.BRACKETLEFT:
            if self.visualize_layer is not None:
                self.visualize_embedding = max(self.visualize_embedding - 1, 0)

    def init_visualize(self):
        for i in range(len(self.filters)):
            self.act_graph[f"conv2d_{i}"] = self.get_layer("conv2d", i)

    def clear_visualize(self):
        self.visualize_layer = None
        self.remove_image("embedding")

    def rollout(self):
        # do standard rollout
        transition = super().rollout()

        if self.visualize_layer is not None:
            # get layer values to visualize
            values = self.act_result[f"conv2d_{self.visualize_layer}"]

            # build image for selected embedding
            _, rows, cols, _ = values.shape
            image = np.zeros((rows, cols, 1))
            flat_values = values.ravel()
            value_min = min(flat_values)
            value_max = max(flat_values)
            value_range = max([0.1, value_max - value_min])
            for y, row in enumerate(values[0]):
                for x, col in enumerate(row):
                    value = col[self.visualize_embedding]
                    image[y][x][0] = int((value - value_min) / value_range * 255)

            # render image
            width, height = self.get_viewer_size()
            self.add_image("embedding", image, x=10, y=10, width=width, height=height)

        return transition
    # ---------------------------------------------------------------------
