import tensorflow as tf
from glearn.policies.policy import Policy


class RandomPolicy(Policy):
    def __init__(self, config, version=None):
        self.mean = config.get("mean", 0)
        self.stddev = config.get("stddev", 1)

        super().__init__(config, version=version)

    def init_model(self):
        # create feed placeholders
        with tf.name_scope('feeds'):
            inputs, _ = self.create_default_feeds()

        output_shape = (tf.shape(inputs)[0], ) + self.output.shape
        predict = tf.random_normal(output_shape, mean=self.mean, stddev=self.stddev,
                                   dtype=self.output.dtype)
        self.set_fetch("predict", predict, ["predict", "evaluate"])

        self.set_fetch("evaluate", tf.no_op(), "optimize")
        self.set_fetch("optimize", tf.no_op(), "optimize")
