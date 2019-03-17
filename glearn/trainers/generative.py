import numpy as np
from glearn.trainers.trainer import Trainer


class GenerativeTrainer(Trainer):
    def __init__(self, config, latent_size=100, fixed_evaluate_latent=False, **kwargs):
        self.latent_size = latent_size
        self.fixed_evaluate_latent = fixed_evaluate_latent

        super().__init__(config, **kwargs)

    def learning_type(self):
        return "unsupervised"

    def build_generator_network(self, name, definition, z=None, reuse=False):
        if z is None:
            # default to latent feed
            z = self.get_or_create_feed("Z", shape=(self.batch_size, self.latent_size))

        return self.build_network(name, definition, z, reuse=reuse)

    def sample_noise(self, batch_size):
        # generate random latent noise
        return np.random.normal(size=(batch_size, self.latent_size))

    def build_trainer(self):
        # evaluate can use fixed noise
        if self.fixed_evaluate_latent:
            epoch_size = self.dataset.get_epoch_size("test")
            self.evaluate_latents = [self.sample_noise(self.batch_size) for i in range(epoch_size)]

    def evaluate(self, experiment_yield):
        # reset fixed evaluate noise
        if self.fixed_evaluate_latent:
            self.evaluate_index = 0

        super().evaluate(experiment_yield)

    def prepare_feeds(self, queries, feed_map):
        # set latent feed, if expected
        if self.has_feed("Z", queries):
            if self.fixed_evaluate_latent and "evaluate" in queries:
                # evaluate uses fixed latent noise
                latent = self.evaluate_latents[self.evaluate_index]
                self.evaluate_index += 1
            else:
                # every run uses random latent noise
                latent = self.sample_noise(self.batch_size)
            feed_map["Z"] = latent

        return super().prepare_feeds(queries, feed_map)
