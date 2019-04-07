from .policy_gradient import PolicyGradientTrainer


class QActorCriticTrainer(PolicyGradientTrainer):
    def __init__(self, config, Q=None, **kwargs):
        self.Q_definition = Q

        super().__init__(config, **kwargs)

    def build_target(self, state, action):
        self.build_Q(state, action)

        td_error = None  # TODO
        self.optimize_Q(td_error)
