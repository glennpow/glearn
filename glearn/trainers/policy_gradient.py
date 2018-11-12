from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def init_optimizer(self):
        # get loss from policy
        loss = self.policy.get_fetch("loss", "evaluate")
        if loss is None:
            self.error(f"Policy ({self.policy}) does not define a 'loss' feed for 'evaluate'")
            return None

        self.summary.add_scalar("loss", loss, "evaluate")

        # minimize policy loss
        self.add_loss(loss, "policy_optimize")
        self.optimize_loss("policy_optimize")

        super().init_optimizer()
