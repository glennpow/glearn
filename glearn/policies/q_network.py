import tensorflow as tf
from glearn.policies.network import NetworkPolicy


class QNetworkPolicy(NetworkPolicy):
    def __init__(self, config, context, network, **kwargs):
        super().__init__(config, context, network, name="Q", **kwargs)

    def handle_predict(self, inputs, predict):
        self.Q = predict

        # build action prediction from Q-values given by network
        with self.variable_scope(self.network.scope):
            predict = tf.argmax(self.Q, axis=-1)

        super().handle_predict(inputs, predict)
