import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def init_optimizer(self):
        # build policy loss
        self.policy.build_loss()

        # get loss from policy
        loss = self.policy.get_fetch("loss", "evaluate")
        if loss is None:
            self.error(f"Policy ({self.policy}) does not define a 'loss' for 'evaluate' graph")
            return None

        self.summary.add_scalar("loss", loss, "evaluate")

        # minimize policy loss
        graph = "policy_optimize"
        with tf.name_scope(graph):
            optimize = self.optimize_loss(loss, graph)

            self.policy.set_fetch(graph, optimize)

        super().init_optimizer()
