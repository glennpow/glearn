import numpy as np
import tensorflow as tf
from .generative import GenerativeTrainer
from glearn.datasets.labeled import LabeledDataset


class VariationalAutoencoderTrainer(GenerativeTrainer):
    def __init__(self, config, encoder=None, decoder=None, **kwargs):
        # get basic params
        self.encoder_definition = encoder
        self.decoder_definition = decoder

        super().__init__(config, **kwargs)

        # only works for labeled datasets
        assert(self.has_dataset)
        assert(isinstance(self.dataset, LabeledDataset))

    def build_encoder(self, x):
        self.encoder_network = self.build_network("encoder", self.encoder_definition, x)
        return self.encoder_network.outputs

    def build_decoder(self, z, reuse=False):
        self.decoder_network = self.build_generator_network("decoder", self.decoder_definition,
                                                            z=z, reuse=reuse)
        # decoded = tf.clip_by_value(decoded, 1e-8, 1 - 1e-8)  # was this needed?
        return self.decoder_network.outputs

    def build_vae_loss(self, x, decoded, optimize=True, loss_name="vae_loss"):
        with tf.variable_scope("vae_optimize"):
            # losses
            encoder_distribution = self.encoder_network.get_distribution_layer()
            mu = encoder_distribution.references["mu"]
            sigma = encoder_distribution.references["sigma"]
            y = decoded
            x = tf.reshape(x, [-1, np.prod(x.shape[1:])])
            marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
            marginal_likelihood = tf.reduce_mean(marginal_likelihood)
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                                tf.log(1e-8 + tf.square(sigma)) - 1, 1)
            KL_divergence = tf.reduce_mean(KL_divergence)
            # KL_divergence = encoder_distribution.kl_divergence(sample_noise) ?
            ELBO = marginal_likelihood - KL_divergence
            loss = -ELBO

            # summaries
            self.add_evaluate_metric("marginal_likelihood", marginal_likelihood)
            self.add_evaluate_metric("kl_divergence", KL_divergence)
            if loss_name:
                self.add_evaluate_metric(loss_name, loss)

            # minimize loss
            if optimize:
                optimize_networks = [self.encoder_network, self.decoder_network]
                self.optimize_loss(loss, networks=optimize_networks, name="vae_optimize")

        return loss

    def build_vae_summary_images(self, x, decoded):
        with tf.variable_scope("summary_images"):
            images = tf.reshape(decoded, tf.shape(x))
            labels = self.get_feed("Y")
            label_count = len(self.dataset.label_names)
            indexes = [tf.where(tf.equal(labels, l))[:, 0] for l in range(label_count)]
            for i in range(label_count):
                label_name = self.dataset.label_names[i]
                index = tf.squeeze(indexes[i])[0]
                decoded_image = images[index:index + 1]
                self.summary.add_images(f"decoded_{label_name}", decoded_image)

    def build_trainer(self):
        with tf.variable_scope("vae"):
            # real input
            x = self.get_feed("X")

            # build encoder-network
            encoded = self.build_encoder(x)

            # build decoder-network
            decoded = self.build_decoder(encoded)

            # build VAE loss
            self.build_vae_loss(x, decoded)

            # generated image summaries
            self.build_vae_summary_images(x, decoded)

    def optimize(self, batch, feed_map):
        # optimize encoder/decoder-networks
        results = self.run("vae_optimize", feed_map)

        return results
