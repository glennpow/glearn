import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.networks import load_network
# from glearn.datasets.labeled import LabeledDataset


class GenerativeAdversarialNetworkTrainer(Trainer):
    def __init__(self, config, policy, discriminator, generator, discriminator_steps=1,
                 noise_size=100, test_size=16, **kwargs):
        # get basic params
        self.discriminator_definition = discriminator
        self.discriminator_steps = discriminator_steps
        self.noise_size = noise_size
        self.test_size = test_size

        super().__init__(config, policy, **kwargs)

        # only works for datasets
        assert(self.has_dataset)
        # assert(isinstance(self.dataset, LabeledDataset))

        # self.policy_scope = "generator"

    def learning_type(self):
        return "unsupervised"

    def build_trainer(self):
        # gather test noise
        self.test_noise = self.sample_noise(self.test_size, self.noise_size)

        with tf.variable_scope("gan"):
            # get original images
            x = self.policy.inputs  # FIXME
            x_shape = tf.shape(x)
            x_size = tf.reduce_prod(x_shape[1:])
            x = tf.reshape(x, [-1, x_size])

            # build decoder-network
            discriminator_input = x
            self.discriminator_network = load_network(f"discriminator", self.policy,
                                                      self.discriminator_definition)
            y = self.discriminator_network.build_predict(discriminator_input)
            # y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
            self.add_fetch("discriminator", y)

            # losses
            policy_distribution = self.policy.network.get_distribution_layer()
            mu = policy_distribution.references["mu"]
            sigma = policy_distribution.references["sigma"]
            marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
            marginal_likelihood = tf.reduce_mean(marginal_likelihood)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                                tf.log(1e-8 + tf.square(sigma)) - 1, 1)
            KL_divergence = tf.reduce_mean(KL_divergence)
            ELBO = marginal_likelihood - KL_divergence
            loss = -ELBO

            # summaries
            self.summary.add_scalar("marginal_likelihood", marginal_likelihood)
            self.summary.add_scalar("KL_divergence", KL_divergence)
            self.summary.add_scalar("ELBO", ELBO)
            self.summary.add_scalar("loss", loss)

            # generated image summaries
            images = tf.reshape(y, x_shape)
            labels = self.get_feed("Y")
            label_count = len(self.dataset.label_names)
            indexes = [tf.where(tf.equal(labels, l))[:, 0] for l in range(label_count)]
            for i in range(label_count):
                label_name = self.dataset.label_names[i]
                index = tf.squeeze(indexes[i])[0]
                decoded_image = images[index:index + 1]
                self.summary.add_image(f"decoded_{label_name}", decoded_image)

            # minimize loss
            learning_rate = self.decoder_definition.get("learning_rate", 1e-3)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            update_op = optimizer.minimize(loss)
            self.add_fetch("vae_update", update_op)

    def sample_noise(rows, cols):
        return np.random.normal(size=(rows, cols))

    def optimize(self, batch, feed_map):
        # optimize discriminator-network
        for i in range(self.discriminator_steps):

            self.run("discriminator_update", feed_map)

        # optimize generator-network
        results = self.run("generator_update", feed_map)

        return results
