import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer


class GenerativeAdversarialNetworkTrainer(Trainer):
    def __init__(self, config, discriminator, generator, discriminator_steps=1,
                 noise_size=100, summary_images=4, fixed_evaluate_noise=False, **kwargs):
        # get basic params
        self.discriminator_definition = discriminator
        self.discriminator_steps = discriminator_steps
        self.generator_definition = generator
        self.noise_size = noise_size
        self.summary_images = summary_images
        self.fixed_evaluate_noise = fixed_evaluate_noise

        super().__init__(config, **kwargs)

        # only works for datasets
        assert(self.has_dataset)

    def learning_type(self):
        return "unsupervised"

    def build_generator(self):
        latent = self.create_feed("latent", shape=(self.batch_size, self.noise_size))
        self.generator_network = self.build_network("generator", self.generator_definition, latent)
        return self.generator_network.outputs

    def build_discriminator(self, generated):
        with tf.name_scope("normalize_images"):
            x = self.get_feed("X")
            x = x * 2 - 1  # normalize (-1, 1)
        self.real_discriminator_network = self.build_network("discriminator",
                                                             self.discriminator_definition, x)
        self.fake_discriminator_network = self.build_network("discriminator",
                                                             self.discriminator_definition,
                                                             generated, reuse=True)
        return self.real_discriminator_network.outputs, self.fake_discriminator_network.outputs

    def build_discriminator_loss(self, optimize=True):
        with tf.variable_scope("discriminator_optimize"):
            real_logits = self.real_discriminator_network.get_output_layer().references["Z"]
            fake_logits = self.fake_discriminator_network.get_output_layer().references["Z"]
            real_labels = tf.ones_like(real_logits)
            D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                  labels=real_labels)
            D_loss_real = tf.reduce_mean(D_loss_real)
            fake_labels = tf.zeros_like(fake_logits)
            D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                  labels=fake_labels)
            D_loss_fake = tf.reduce_mean(D_loss_fake)
            D_loss = D_loss_real + D_loss_fake

            self.add_fetch("discriminator_loss", D_loss, "evaluate")
            self.summary.add_scalar("loss", D_loss)

            if optimize:
                self.add_fetch("discriminator_optimize",
                               self.real_discriminator_network.optimize_loss(D_loss))

    def build_generator_loss(self, optimize=True):
        with tf.variable_scope("generator_optimize"):
            fake_logits = self.fake_discriminator_network.get_output_layer().references["Z"]
            G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                             labels=tf.ones_like(fake_logits))
            G_loss = tf.reduce_mean(G_loss)

            self.add_fetch("generator_loss", G_loss, "evaluate")
            self.summary.add_scalar("loss", G_loss)

            if optimize:
                self.add_fetch("generator_optimize", self.generator_network.optimize_loss(G_loss))

    def build_gan_summary_images(self, generated):
        with tf.variable_scope("summary_images"):
            # original image summaries
            original_images = self.get_feed("X")
            self.summary.add_images(f"real", original_images, self.summary_images)

            # generated image summaries
            generated_images = tf.reshape(generated, tf.shape(original_images))
            self.summary.add_images(f"generated", generated_images, self.summary_images)

    def build_trainer(self):
        # evaluate can use fixed noise
        if self.fixed_evaluate_noise:
            epoch_size = self.dataset.get_epoch_size("test")
            self.evaluate_latents = [self.sample_noise(self.batch_size, self.noise_size)
                                     for i in range(epoch_size)]

        with tf.variable_scope("gan"):
            # build generator-network
            generated = self.build_generator()

            # build discriminator-networks
            self.build_discriminator(generated)

            # optimize discriminator loss
            self.build_discriminator_loss()

            # optimize generator loss
            self.build_generator_loss()

            # summary images
            self.build_gan_summary_images(generated)

    def sample_noise(self, rows, cols):
        # generate a random latent noise sample
        return np.random.normal(size=(rows, cols))

    def evaluate(self, experiment_yield):
        # reset fixed evaluate noise
        if self.fixed_evaluate_noise:
            self.evaluate_index = 0

        super().evaluate(experiment_yield)

    def prepare_feeds(self, queries, feed_map):
        if self.fixed_evaluate_noise and "evaluate" in queries:
            # evaluate uses fixed latent noise
            latent = self.evaluate_latents[self.evaluate_index]
            self.evaluate_index += 1
        else:
            # every run uses random latent noise
            latent = self.sample_noise(self.batch_size, self.noise_size)
        feed_map["latent"] = latent

        return super().prepare_feeds(queries, feed_map)

    def optimize(self, batch, feed_map):
        # optimize discriminator-network
        for i in range(self.discriminator_steps):
            D_feed_map = {
                "X": feed_map["X"],
            }
            self.run("discriminator_optimize", D_feed_map)

        # optimize generator-network
        results = self.run("generator_optimize", {})

        return results
