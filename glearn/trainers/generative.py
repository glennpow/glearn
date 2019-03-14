import numpy as np
from glearn.trainers.trainer import Trainer


class GenerativeTrainer(Trainer):
    def __init__(self, config, latent_size=100, **kwargs):
        self.latent_size = latent_size

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
