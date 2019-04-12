import numpy as np
import tensorflow as tf
from .generative import GenerativeTrainer


class VariationalAutoencoderTrainer(GenerativeTrainer):
    EPSILON = 1e-5

    def __init__(self, config, encoder=None, decoder=None, **kwargs):
        # get basic params
        self.encoder_definition = encoder
        self.decoder_definition = decoder

        super().__init__(config, **kwargs)

    def build_encoder(self, x, y):
        # condition input (TODO - inject properly into first dense layer)
        if self.conditional:
            x = self.get_conditioned_inputs(x, y)

        self.encoder_network = self.build_network("encoder", self.encoder_definition, x)
        return self.encoder_network.outputs

    def build_decoder(self, z, reuse=False):
        self.decoder_network = self.build_generator_network("decoder", self.decoder_definition,
                                                            z=z, reuse=reuse)
        return self.decoder_network.outputs

    def build_kl_divergence(self):
        encoder_distribution = self.encoder_network.get_distribution_layer()
        mu = encoder_distribution.references["mu"]
        sigma = encoder_distribution.references["sigma"]
        kl_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                            tf.log(self.EPSILON + tf.square(sigma)) - 1, 1)
        kl_divergence = tf.reduce_mean(kl_divergence)

        # summary
        self.add_metric("KL_divergence", kl_divergence)

        return kl_divergence

    def build_vae_optimize(self, x, decoded, loss_name="VAE_loss"):
        with tf.variable_scope("VAE_optimize"):
            with tf.variable_scope("loss"):
                # losses
                y = decoded
                y = tf.clip_by_value(y, self.EPSILON, 1 - self.EPSILON)
                x = tf.reshape(x, [-1, np.prod(x.shape[1:])])
                marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
                marginal_likelihood = tf.reduce_mean(marginal_likelihood)
                kl_divergence = self.build_kl_divergence()
                elbo = marginal_likelihood - kl_divergence
                loss = -elbo

            # summaries
            self.add_metric("marginal_likelihood", marginal_likelihood)
            if loss_name:
                self.add_metric(loss_name, loss)

            # minimize loss
            optimize_networks = [self.encoder_network, self.decoder_network]
            self.optimize_loss(loss, networks=optimize_networks, name="VAE_optimize")

        return loss

    def build_trainer(self):
        super().build_trainer()

        with tf.variable_scope("VAE"):
            # real images and labels
            x = self.get_feed("X")
            y = self.get_feed("Y")

            # build encoder-network
            encoded = self.build_encoder(x, y)

            # build decoder-network
            decoded = self.build_decoder(encoded)

            # build VAE loss
            self.build_vae_optimize(x, decoded)

            # generated image summaries
            self.build_summary_images("decoded", decoded, labels=y)

    def optimize(self, batch):
        feed_map = batch.get_feeds()

        # optimize encoder/decoder-networks
        results = self.run("VAE_optimize", feed_map)

        return results
