import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.datasets.labeled import LabeledDataset


class GenerativeTrainer(Trainer):
    def __init__(self, config, latent_size=100, conditional_latent=False,
                 fixed_evaluate_latent=False, **kwargs):
        self.latent_size = latent_size
        self.conditional_latent = conditional_latent
        self.fixed_evaluate_latent = fixed_evaluate_latent

        super().__init__(config, **kwargs)

        assert(self.has_dataset)

        self.has_labeled_dataset = isinstance(self.dataset, LabeledDataset)

    def learning_type(self):
        return "unsupervised"

    def get_latent_size(self):
        if self.conditional_latent and self.has_labeled_dataset:
            return self.latent_size  # TODO
        else:
            return self.latent_size

    def build_generator_network(self, name, definition, z=None, reuse=False):
        if z is None:
            # default to latent feed
            z = self.get_or_create_feed("Z", shape=(self.batch_size, self.get_latent_size()))

        return self.build_network(name, definition, z, reuse=reuse)

    def sample_noise(self, batch_size):
        # generate random latent noise
        return np.random.normal(size=(batch_size, self.latent_size))

    def build_trainer(self):
        # evaluate can use fixed noise
        if self.fixed_evaluate_latent:
            epoch_size = self.dataset.get_epoch_size("test")
            self.evaluate_latents = [self.sample_noise(self.batch_size) for i in range(epoch_size)]

    def build_summary_images(self, name, images, labels=None):
        # image summaries
        with tf.variable_scope("summary_images"):
            images = tf.reshape(images, tf.shape(self.get_feed("X")))

            if labels is not None and self.has_labeled_dataset:
                label_count = len(self.dataset.label_names)
                indexes = [tf.where(tf.equal(labels, l))[:, 0] for l in range(label_count)]
                for i in range(label_count):
                    label_name = self.dataset.label_names[i]
                    index = tf.squeeze(indexes[i])[0]
                    decoded_image = images[index:index + 1]
                    self.summary.add_images(f"{name}_{label_name}", decoded_image)
            else:
                self.summary.add_images(name, images, self.summary_images)

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
