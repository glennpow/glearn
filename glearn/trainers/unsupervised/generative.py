import numpy as np
import tensorflow as tf
from .unsupervised import UnsupervisedTrainer
from glearn.datasets.labeled import LabeledDataset


class GenerativeTrainer(UnsupervisedTrainer):
    def __init__(self, config, latent_size=100, conditional=False, fixed_evaluate_latent=False,
                 **kwargs):
        self.latent_size = latent_size
        self.conditional = conditional
        self.fixed_evaluate_latent = fixed_evaluate_latent

        super().__init__(config, **kwargs)

        assert(self.has_dataset)

        self.has_labeled_dataset = isinstance(self.dataset, LabeledDataset)
        self.conditional = self.conditional and self.has_labeled_dataset

    def learning_type(self):
        return "unsupervised"

    def get_conditioned_inputs(self, inputs, labels):
        # TODO - investigate: condition_tensor_from_onehot() (https://arxiv.org/abs/1609.03499)
        return tf.concat([inputs, tf.cast(labels, tf.float32)], -1)

    def build_generator_network(self, name, definition, z=None, reuse=False):
        if z is None:
            # default to latent feed
            z = self.get_or_create_feed("Z", shape=(self.batch_size, self.latent_size))

        # conditional inputs
        if self.conditional:
            # TODO - investigate: condition_tensor_from_onehot() (https://arxiv.org/abs/1609.03499)
            z = self.get_conditioned_inputs(z, self.get_feed("Y"))

        return self.build_network(name, definition, z, reuse=reuse)

    def sample_noise(self, batch_size):
        # generate random latent noise  (spherical gaussian)
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

    def evaluate(self):
        # reset fixed evaluate noise
        if self.fixed_evaluate_latent:
            self.evaluate_index = 0

        super().evaluate()

    def prepare_feeds(self, query, feed_map):
        super().prepare_feeds(query, feed_map)

        # set latent feed, if expected
        if self.has_feed("Z", query):
            if self.fixed_evaluate_latent and "evaluate" in query:
                # evaluate uses fixed latent noise
                latent = self.evaluate_latents[self.evaluate_index]
                self.evaluate_index += 1
            else:
                # every run uses random latent noise
                latent = self.sample_noise(self.batch_size)
            feed_map["Z"] = latent
