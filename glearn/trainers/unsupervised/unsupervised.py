from glearn.trainers.supervised.supervised import SupervisedTrainer


class UnsupervisedTrainer(SupervisedTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def learning_type(self):
        return "unsupervised"
