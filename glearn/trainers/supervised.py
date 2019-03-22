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

    def get_batch(self, mode="train"):
        # dataset batch of samples
        return self.dataset.get_batch(mode=mode)

    def experiment_loop(self):
        # supervised training loop
        if self.training:
            # train desired epochs
            self.epoch = 0

            while self.epochs is None or self.epoch < self.epochs:
                # start current epoch
                self.epoch += 1
                self.epoch_step = 0
                self.epoch_start_time = self.time()
                epoch_steps = self.reset()

                # epoch summary
                global_epoch = self.current_global_step / epoch_steps
                self.summary.add_simple_value("epoch", global_epoch)

                for step in range(epoch_steps):
                    # epoch time
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
            # evaluate single epoch
            self.epoch = 1
            self.epoch_start_time = self.time()
            self.evaluate_and_report()
