import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def build_trainer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # build policy loss
            loss, _ = self.policy.build_loss(self.get_feed("Y"))

            # minimize policy loss
            self.policy.optimize_loss(loss, name=query)
