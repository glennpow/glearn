import numpy as np
import tensorflow as tf
from .generative import GenerativeTrainer


class VariationalAutoencoderTrainer(GenerativeTrainer):
    def __init__(self, config, encoder=None, decoder=None, **kwargs):
        # get basic params
        self.encoder_definition = encoder
        self.decoder_definition = decoder

        super().__init__(config, **kwargs)

    def build_encoder(self, x):
        self.encoder_network = self.build_network("encoder", self.encoder_definition, x)
        return self.encoder_network.outputs

    def build_decoder(self, z, reuse=False):
        self.decoder_network = self.build_generator_network("decoder", self.decoder_definition,
                                                            z=z, reuse=reuse)
        # decoded = tf.clip_by_value(decoded, 1e-8, 1 - 1e-8)  # was this needed?
        return self.decoder_network.outputs

    def build_kl_divergence(self):
        encoder_distribution = self.encoder_network.get_distribution_layer()
        mu = encoder_distribution.references["mu"]
        sigma = encoder_distribution.references["sigma"]
        kl_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                            tf.log(1e-8 + tf.square(sigma)) - 1, 1)
        kl_divergence = tf.reduce_mean(kl_divergence)

        # summary
        self.add_evaluate_metric("kl_divergence", kl_divergence)

        return kl_divergence
        # return encoder_distribution.kl_divergence(sample_noise) ?

    def build_vae_loss(self, x, decoded, loss_name="vae_loss"):
        with tf.variable_scope("vae_optimize"):
            # losses
            y = decoded
            x = tf.reshape(x, [-1, np.prod(x.shape[1:])])
            marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
            marginal_likelihood = tf.reduce_mean(marginal_likelihood)
            kl_divergence = self.build_kl_divergence()
            elbo = marginal_likelihood - kl_divergence
            loss = -elbo

            # summaries
            self.add_evaluate_metric("marginal_likelihood", marginal_likelihood)
            if loss_name:
                self.add_evaluate_metric(loss_name, loss)

            # minimize loss
            optimize_networks = [self.encoder_network, self.decoder_network]
            self.optimize_loss(loss, networks=optimize_networks, name="vae_optimize")

        return loss

    def build_trainer(self):
        super().build_trainer()

        with tf.variable_scope("vae"):
            # real images and labels
            x = self.get_feed("X")
            y = self.get_feed("Y")

            # build encoder-network
            encoded = self.build_encoder(x)

            # build decoder-network
            decoded = self.build_decoder(encoded)

            # build VAE loss
            self.build_vae_loss(x, decoded)

            # generated image summaries
            self.build_summary_images("decoded", decoded, labels=y)

    def optimize(self, batch, feed_map):
        # optimize encoder/decoder-networks
        results = self.run("vae_optimize", feed_map)

        return results
