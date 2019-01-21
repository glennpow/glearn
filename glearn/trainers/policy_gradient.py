import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def init_optimizer(self):
        query = "policy_optimize"
        with tf.name_scope(query):
            # build policy loss
            self.policy.build_loss()

            # get loss from policy
            loss = self.policy.get_fetch("loss", "evaluate")
            if loss is None:
                self.error(f"Policy ({self.policy}) doesn't define a 'loss' for 'evaluate' query")
                return None

            # minimize policy loss
            optimize = self.optimize_loss(loss, query)

            self.policy.set_fetch(query, optimize)

            self.summary.add_scalar("total_loss", loss, "evaluate")
