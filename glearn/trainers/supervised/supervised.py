import time
from glearn.trainers.trainer import Trainer


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

    def get_batch(self, mode="train"):
        # dataset batch of samples
        return self.dataset.get_batch(mode=mode)

    def experiment_loop(self):
        # supervised training loop
        self.epoch = 0

        while not self.training or self.epochs is None or self.epoch < self.epochs:
            # start current epoch
            self.epoch += 1
            self.epoch_step = 0
            self.epoch_start_time = time.time()

            if self.training:
                # current epoch summary
                epoch_steps = self.reset(mode="train" if self.training else "test")
                global_epoch = self.current_global_step / epoch_steps
                self.summary.add_simple_value("global_epoch", global_epoch)

                for step in range(epoch_steps):
                    # epoch step
                    self.epoch_step = step + 1

                    # optimize batch
                    self.batch = self.get_batch()
                    self.optimize_and_report(self.batch)

                    # evaluate if time to do so
                    if self.should_evaluate():
                        self.evaluate_and_report()

                    if self.experiment_yield(True):
                        return
            else:
                # current epoch summary
                self.summary.add_simple_value("epoch", self.epoch)

                # evaluate single epoch
                self.evaluate_and_report()

                if self.experiment_yield(True):
                    return
