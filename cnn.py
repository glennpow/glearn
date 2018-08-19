import numpy as np
import tensorflow as tf
import pyglet
from policy import Policy
from layers import add_fc, add_conv2d


class CNN(Policy):
    # filters: [height, width, chan_in, chan_out]
    def __init__(self, env=None, dataset=None, batch_size=128, seed=0, learning_rate=0.001,
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
        self.act_op = tf.nn.softmax(logits, name='act')

        # calculate loss
        with tf.name_scope('loss'):
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            # loss = tf.reduce_mean(neg_log_p * self.discounted_rewards)  # TODO?
            loss = tf.reduce_mean(neg_log_p)

        # minimize loss
        with tf.name_scope('optimize'):
            self.optimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    # HACKs - to view embeddings
    # ---------------------------------------------------------------------
    embedding = None

    def init_viewer(self):
        super().init_viewer()

        if self.viewer is not None:
            self.viewer.window.push_handlers(self)

    def on_key_press(self, key, modifiers):
        if key == pyglet.window.key._0:
            self.embedding = None
            self.remove_image("embedding")
        elif key == pyglet.window.key._1:
            self.embedding = 1
        elif key == pyglet.window.key._2:
            self.embedding = 2
        elif key == pyglet.window.key._3:
            self.embedding = 3
        elif key == pyglet.window.key._4:
            self.embedding = 4
        elif key == pyglet.window.key._5:
            self.embedding = 5
        elif key == pyglet.window.key._6:
            self.embedding = 6
        elif key == pyglet.window.key._7:
            self.embedding = 7
        elif key == pyglet.window.key._8:
            self.embedding = 8
        elif key == pyglet.window.key._9:
            self.embedding = 9

    def rollout(self):
        # do standard rollout
        transition = super().rollout()

        # get layer to view embeddings
        layer_index = 0
        layer = self.get_layer("conv2d", layer_index)

        # get layer embedding values
        observation = transition.observation
        feed_dict = self.act_inputs(observation, {self.inputs: [observation]})
        values = self.sess.run(layer, feed_dict=feed_dict)

        _, rows, cols, embeddings = values.shape
        embedding = self.embedding
        if embedding is not None:
            # build image for selected embedding
            assert embedding < embeddings
            image = np.zeros((rows, cols, 1))
            flat_values = values.ravel()
            value_min = min(flat_values)
            value_max = max(flat_values)
            value_range = max([0.1, value_max - value_min])
            for y, row in enumerate(values[0]):
                for x, col in enumerate(row):
                    value = col[embedding]
                    image[y][x][0] = int((value - value_min) / value_range * 255)

            # render image
            width, height = self.get_viewer_size()
            self.add_image("embedding", image, x=10, y=10, width=width, height=height)

        return transition
    # ---------------------------------------------------------------------

    def act_inputs(self, observation, feed_dict):
        # no dropout for inference
        feed_dict[self.dropout] = 1

        return feed_dict

    def optimize_inputs(self, batch, feed_dict):
        feed_dict[self.dropout] = self.dropout_rate

        return feed_dict
