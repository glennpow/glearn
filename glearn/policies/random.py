# import tensorflow as tf
from glearn.policies.policy import Policy


class RandomPolicy(Policy):
    # def init_model(self):
    #     # create feed placeholders
    #     with tf.name_scope('feeds'):
    #         inputs, _ = self.create_default_feeds()

    #     output_shape = (tf.shape(inputs)[0], ) + self.output.shape
    #     dtype = self.output.dtype
    #     predict = tf.random_uniform(output_shape, minval=self.output.low, ..., dtype=dtype)
    #     self.set_fetch("predict", predict, ["predict", "evaluate"])

    #     self.set_fetch("evaluate", tf.no_op(), "optimize")

    def run(self, sess, graph, feed_map, **kwargs):
        output = []
        for i in range(len(feed_map["X"])):
            output.append(self.output.sample())
        return {graph: output}
