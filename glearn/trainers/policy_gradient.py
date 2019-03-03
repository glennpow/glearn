import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # build policy loss
            self.policy.build_loss()

            # minimize policy loss
            optimize = self.policy.optimize_loss()
            self.add_fetch(query, optimize)
