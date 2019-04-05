from glearn.trainers.policy_gradient import PolicyGradientTrainer


class AdvantageActorCriticTrainer(PolicyGradientTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        assert self.V_definition is not None

    def get_baseline_scope(self):
        return "critic"
