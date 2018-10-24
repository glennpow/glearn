from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def init_optimizer(self):
        # minimize policy loss
        self.optimize_loss("policy_optimize")

        super().init_optimizer()
