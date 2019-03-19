import time
from glearn.trainers import Trainer


class SupervisedTrainer(Trainer):
    def __init__(self, config, epochs=None, **kwargs):
        super().__init__(config, **kwargs)

        self.epochs = epochs

    def get_info(self):
        info = super().get_info()
        info.update({
            "Epochs": self.epochs,
        })
        return info

    def learning_type(self):
        return "supervised"

    def reset(self, mode="train"):
        return self.dataset.reset(mode=mode)

    def get_iteration_name(self):
        return "Epoch"

    def get_batch(self, mode="train"):
        # dataset batch of samples
        return self.dataset.get_batch(mode=mode)

    def experiment_loop(self):
        # dataset learning
        if self.training:
            # train desired epochs
            self.iteration = 0

            while self.epochs is None or self.iteration < self.epochs:
                # start current epoch
                self.iteration += 1
                self.iteration_step = 0
                self.iteration_start_time = time.time()
                epoch_steps = self.reset()

                # epoch summary
                global_epoch = self.current_global_step / epoch_steps
                self.summary.add_simple_value("epoch", global_epoch, "experiment")

                for step in range(epoch_steps):
                    # epoch time
                    self.iteration_step = step + 1

                    # optimize batch
                    self.batch, feed_map = self.get_batch()
                    self.optimize_and_report(self.batch, feed_map)

                    # evaluate if time to do so
                    if self.should_evaluate():
                        self.evaluate()

                    if self.experiment_yield(True):
                        return
        else:
            # evaluate single epoch
            self.epoch = 0
            self.iteration_start_time = time.time()
            self.evaluate()
