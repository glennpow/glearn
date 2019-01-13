import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def init_optimizer(self):
        # build policy loss
        self.policy.build_loss()

        # get loss from policy
        loss = self.policy.get_fetch("loss", "evaluate")
        if loss is None:
            self.error(f"Policy ({self.policy}) does not define a 'loss' for 'evaluate' family")
            return None

        self.summary.add_scalar("loss", loss, "evaluate")

        # minimize policy loss
        family = "policy_optimize"
        with tf.name_scope(family):
            optimize = self.optimize_loss(loss, family)

            self.policy.set_fetch(family, optimize)

        super().init_optimizer()
