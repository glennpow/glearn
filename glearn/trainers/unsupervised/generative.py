import numpy as np
import tensorflow as tf
from .unsupervised import UnsupervisedTrainer
from glearn.datasets.labeled import LabeledDataset


class GenerativeTrainer(UnsupervisedTrainer):
    def __init__(self, config, latent_size=100, conditional=False, fixed_evaluate_latent=False,
                 summary_images=9, **kwargs):
        self.latent_size = latent_size
        self.conditional = conditional
        self.fixed_evaluate_latent = fixed_evaluate_latent
        self.summary_images = summary_images

        super().__init__(config, **kwargs)

        assert(self.has_dataset)

        # TODO - could use partial images as conditions as well...
        self.has_labeled_dataset = isinstance(self.dataset, LabeledDataset)
        self.conditional = self.conditional and self.has_labeled_dataset

    def learning_type(self):
        return "unsupervised"

    def get_conditioned_inputs(self, inputs, labels):
        return tf.contrib.gan.features.condition_tensor_from_onehot(inputs, labels)
        # return tf.concat([tf.reshape(inputs, (-1, np.prod(inputs.shape[1:]))),
        #                   tf.expand_dims(tf.cast(labels, tf.float32), -1)], -1)
        # return tf.concat([inputs, tf.cast(labels, tf.float32)], -1)

    def build_generator_network(self, name, definition, z=None, reuse=None):
        if z is None:
            # default to latent feed
            z = self.get_or_create_feed("Z", shape=(self.batch_size, self.latent_size))

        # conditional inputs
        if self.conditional:
            with self.variable_scope(f"{name}_network") as _:
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
        with self.variable_scope("summary_images"):
            images = tf.reshape(images, tf.shape(self.get_feed("X")))

            if labels is not None and self.has_labeled_dataset:
                label_count = len(self.dataset.label_names)
                # indexes = [tf.where(tf.equal(labels, lb)) for lb in range(label_count)]
                for i in range(label_count):
                    label = self.dataset.labels[i]
                    label_name = self.dataset.label_names[i]

                    if self.dataset.one_hot:
                        # handle one hot labels
                        # label_one_hot = tf.cast(tf.one_hot(i, label_count), tf.int8)
                        index = tf.where(tf.reduce_all(tf.equal(labels, label), axis=-1))[0]
                        index = tf.squeeze(index)
                    else:
                        # handle int labels
                        index = tf.squeeze(tf.where(tf.equal(labels, i)))[0]
                        # index = tf.squeeze(indexes[i])[0]  # TODO - make sure at least one

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
