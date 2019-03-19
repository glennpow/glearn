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

    def get_iteration_name(self):
        return "Epoch"

    def get_batch(self, mode="train"):
        # dataset batch of samples
        return self.dataset.get_batch(mode=mode)

    def reset_evaluate(self):
        return self.dataset.reset(mode="test")

    def experiment_loop(self):
        # dataset learning
        if self.training:
            # train desired epochs
            self.iteration = 1

            while self.epochs is None or self.iteration <= self.epochs:
                # start current epoch
                epoch_size = self.dataset.reset(mode="train")
                self.iteration_step = 0
                self.iteration_start_time = time.time()

                # epoch summary
                global_epoch = self.current_global_step / epoch_size
                self.summary.add_simple_value("epoch", global_epoch, "experiment")

                for step in range(epoch_size):
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

                self.iteration += 1
        else:
            # evaluate single epoch
            self.epoch = 0
            self.iteration_start_time = time.time()
            self.evaluate()
