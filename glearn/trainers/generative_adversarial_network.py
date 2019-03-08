import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer


class GenerativeAdversarialNetworkTrainer(Trainer):
    def __init__(self, config, discriminator, generator, discriminator_steps=1,
                 generator_samples=4, noise_size=100, test_size=16, **kwargs):
        # get basic params
        self.discriminator_definition = discriminator
        self.discriminator_steps = discriminator_steps
        self.generator_definition = generator
        self.generator_samples = generator_samples
        self.noise_size = noise_size
        self.test_size = test_size

        super().__init__(config, **kwargs)

        # only works for datasets
        assert(self.has_dataset)

    def learning_type(self):
        return "unsupervised"

    def build_trainer(self):
        with tf.variable_scope("gan"):
            # build generator-network
            latent = self.create_feed("latent", shape=(self.batch_size, self.noise_size))
            G_network = self.build_network("generator", self.generator_definition, latent)
            G = G_network.outputs

            # build discriminator-networks
            with tf.variable_scope("real_images"):
                x = self.get_feed("X")
                x_shape = tf.shape(x)
                x_size = np.prod(x.shape[1:])
                x = tf.reshape(x, (-1, x_size))
                x = x * 2 - 1  # normalize (-1, 1)
            D_real_network = self.build_network("discriminator", self.discriminator_definition, x)
            D_real = D_real_network.get_output_layer().references["Z"]
            D_fake_network = self.build_network("discriminator", self.discriminator_definition, G,
                                                reuse=True)
            D_fake = D_fake_network.get_output_layer().references["Z"]

            # optimize discriminator loss
            with tf.variable_scope("discriminator_loss"):
                D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,
                                                                      labels=tf.ones_like(D_real))
                D_loss_real = tf.reduce_mean(D_loss_real)
                D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                      labels=tf.zeros_like(D_fake))
                D_loss_fake = tf.reduce_mean(D_loss_fake)
                D_loss = D_loss_real + D_loss_fake
                self.add_fetch("discriminator_optimize", D_real_network.optimize_loss(D_loss))
                self.add_fetch("discriminator_loss", D_loss, "evaluate")

            # optimize generator loss
            with tf.variable_scope("generator_loss"):
                G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                 labels=tf.ones_like(D_fake))
                G_loss = tf.reduce_mean(G_loss)
                self.add_fetch("generator_optimize", G_network.optimize_loss(G_loss))
                self.add_fetch("generator_loss", G_loss, "evaluate")

            # summaries
            self.summary.add_scalar("discriminator_loss", D_loss)
            self.summary.add_scalar("generator_loss", G_loss)

            with tf.variable_scope("summary_images"):
                # original image summaries
                original_images = self.get_feed("X")
                self.summary.add_images(f"real", original_images, self.generator_samples)

                # generated image summaries
                generated_images = tf.reshape(G, x_shape)
                self.summary.add_images(f"generated", generated_images, self.generator_samples)

    def sample_noise(self, rows, cols):
        return np.random.normal(size=(rows, cols))

    def prepare_feeds(self, queries, feed_map):
        feed_map["latent"] = self.sample_noise(self.batch_size, self.noise_size)

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
