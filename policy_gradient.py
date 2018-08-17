import numpy as np
import tensorflow as tf
from policy import Policy


class PolicyGradient(Policy):
    def __init__(self, env, learning_rate=0.01, discount_factor=0.95,
                 hidden_depth=2, hidden_size=10, load_path=None, save_path=None,
                 tensorboard_path=None):
        self.learning_rate = learning_rate  # lamdba λ
        self.discount_factor = discount_factor  # gamma γ
        self.hidden_depth = hidden_depth
        self.hidden_size = hidden_size

        super().__init__(env=env, load_path=load_path, save_path=save_path,
                         tensorboard_path=tensorboard_path)

    def init_model(self):
        # create placeholders
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name="X")
            self.outputs = tf.placeholder(tf.float32, shape=(None,) + self.output_shape, name="Y")
            self.discounted_rewards = tf.placeholder(tf.float32, (None, ), name="V")

        # create hidden layers
        input_size = np.prod(self.input_shape)
        inputs = tf.reshape(self.inputs, (-1, input_size))
        layer = (inputs, None, input_size)
        for i in range(self.hidden_depth):
            layer = self.add_layer(i + 1, layer, self.hidden_size)

        # create output layer
        layer = self.add_layer(self.hidden_depth + 1, layer, np.prod(self.output_shape),
                               activation=tf.nn.softmax)

        # softmax
        logits = layer[1]
        labels = self.outputs
        self.act_op = tf.nn.softmax(logits, name='act')

        # calculate loss
        with tf.name_scope('loss'):
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_p * self.discounted_rewards)

        # minimize loss
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def add_layer(self, num, input_layer, output_size, initializer=None, activation=tf.nn.relu):
        # initializer
        initializer_seed = 1
        if isinstance(initializer, int):
            initializer_seed = initializer
            initializer = None
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer(seed=initializer_seed)

        # create variables and operations
        with tf.name_scope(f"layer_{num}"):
            with tf.variable_scope(f"layer_{num}"):
                input_size = input_layer[2]
                W = tf.get_variable("W", (input_size, output_size), initializer=initializer)
                b = tf.get_variable("b", (output_size, ), initializer=initializer)

            inputs = input_layer[0]
            # x = tf.reshape(inputs, (-1, input_size))
            Z = tf.add(tf.matmul(inputs, W), b)
            A = activation(Z)
            return (A, Z, output_size)

    def train_inputs(self, batch, feed_dict):
        # normalized discount rewards
        discounted_rewards = np.zeros_like(batch.rewards)
        accum = 0
        for t in reversed(range(len(batch.rewards))):
            accum = accum * self.discount_factor + batch.rewards[t]
            discounted_rewards[t] = accum
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        feed_dict[self.discounted_rewards] = discounted_rewards

        return feed_dict
