import numpy as np
import tensorflow as tf
from glearn.policies.policy import Policy
from glearn.policies.layers import add_fc


class PolicyGradientPolicy(Policy):
    def __init__(self, config, version=None):
        self.learning_rate = config.get("learning_rate", 2e-4)  # lamdba λ
        self.discount_factor = config.get("discount_factor", 0.95)  # gamma γ
        self.hidden_depth = config.get("hidden_depth", 2)
        self.hidden_size = config.get("hidden_size", 10)

        super().__init__(config, version=version)

    def init_model(self):
        # create feed placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

            discounted_rewards = tf.placeholder(tf.float32, (None, ), name="gamma")
            self.set_feed("gamma", discounted_rewards, ["optimize", "evaluate"])

        # prepare inputs
        input_size = self.input.size  # FIXME - can we infer this from inputs?
        inputs = tf.reshape(inputs, (-1, input_size))
        layer = tf.cast(inputs, tf.float32)

        # create hidden layers
        for i in range(self.hidden_depth):
            layer, info = add_fc(self, layer, input_size, self.hidden_size)
            input_size = layer.shape[1]

        # create output layer
        layer, info = add_fc(self, layer, input_size, self.output.size, activation=tf.nn.softmax)

        # store prediction
        self.set_fetch("predict", layer, ["predict", "evaluate"])

        # calculate loss
        with tf.name_scope('loss'):
            logits = info["Z"]
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
            loss = tf.reduce_mean(neg_log_p * discounted_rewards)
            self.set_fetch("loss", loss, "evaluate")

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.set_fetch("optimize", optimizer.minimize(loss), "optimize")

    def prepare_feed_map(self, graph, data, feed_map):
        if graph == "optimize" or graph == "evaluate":
            # normalized discount rewards
            rewards = [e.reward for e in data.info["transitions"]]
            discounted_rewards = np.zeros_like(rewards)
            accum = 0
            for t in reversed(range(len(rewards))):
                accum = accum * self.discount_factor + rewards[t]
                discounted_rewards[t] = accum
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            feed_map["gamma"] = discounted_rewards

        return feed_map
