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
        # gather test noise
        self.test_noise = self.sample_noise(self.test_size, self.noise_size)

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
            D_real_network = self.build_network("discriminator", self.discriminator_definition, x)
            D_real = D_real_network.outputs
            D_fake_network = self.build_network("discriminator", self.discriminator_definition, G,
                                                reuse=True)
            D_fake = D_fake_network.outputs

            # optimize discriminator loss
            with tf.variable_scope("losses"):
                D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,
                                                                      labels=tf.ones_like(D_real))
                D_loss_real = tf.reduce_mean(D_loss_real)
                D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                      labels=tf.zeros_like(D_fake))
                D_loss_fake = tf.reduce_mean(D_loss_fake)
                D_loss = D_loss_real + D_loss_fake
                self.add_fetch("discriminator_optimize", D_real_network.optimize_loss(D_loss))
                self.add_fetch("D_loss", D_loss, "evaluate")

                # optimize generator loss
                G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                 labels=tf.ones_like(D_fake))
                G_loss = tf.reduce_mean(G_loss)
                self.add_fetch("generator_optimize", G_network.optimize_loss(G_loss))
                self.add_fetch("G_loss", G_loss, "evaluate")

            # summaries
            self.summary.add_scalar("D_loss", D_loss)
            self.summary.add_scalar("G_loss", G_loss)

            # # generated image summaries
            generated_images = tf.reshape(G, x_shape)
            for i in range(self.generator_samples):
                generated_image = generated_images[i:i + 1]
                self.summary.add_image(f"generated_{i}", generated_image)

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
