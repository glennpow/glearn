import tensorflow as tf
from .variational_autoencoder import VariationalAutoencoderTrainer
from .generative_adversarial_network import GenerativeAdversarialNetworkTrainer


class VAEGANTrainer(VariationalAutoencoderTrainer, GenerativeAdversarialNetworkTrainer):
    def __init__(self, config, gamma=0.5, **kwargs):
        self.gamma = gamma

        VariationalAutoencoderTrainer.__init__(self, config, **kwargs)
        GenerativeAdversarialNetworkTrainer.__init__(self, config, **kwargs)

    def build_trainer(self):
        with tf.variable_scope("VAE_GAN"):
            # get normalized real images and labels
            with tf.name_scope("normalize_images"):
                x = self.get_feed("X")
                x = x * 2 - 1  # normalize (-1, 1)
                y = self.get_feed("Y")

            # build encoder-network
            z = self.build_encoder(x, y)

            # build decoder-network
            decoder_network = self.build_generator_network("decoder", self.decoder_definition, z=z)
            x_tilde = decoder_network.outputs

            # build generator-network
            decoder_p_network = self.build_generator_network("decoder", self.decoder_definition,
                                                             reuse=True)
            x_p = decoder_p_network.outputs
            # x_p = x_p * 2 - 1  # normalize (-1, 1)  # HACK - not needed with tanh act?

            # build discriminator-networks
            self.build_discriminator(x, True)
            self.build_discriminator(x_tilde, False)
            self.build_discriminator(x_p, False)

            # optimize discriminator with loss
            gan_loss = self.build_discriminator_loss()

            # calculate reconstruction VAE loss
            with tf.name_scope("VAE_optimize"):
                l_prior = self.build_kl_divergence()
                d_x = self.discriminator_networks[0]
                d_x_l = d_x.get_layer(-2).references["Z"]
                d_x_tilde = self.discriminator_networks[1]
                d_x_tilde_l = d_x_tilde.get_layer(-2).references["Z"]
                vae_loss = tf.reduce_mean(tf.squared_difference(d_x_l, d_x_tilde_l))  # FIXME?
                # vae_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=?, labels=?)
                # vae_loss += tf.reduce_mean(vae_loss)

            # optimize encoder/decoder
            with tf.name_scope("encoder_optimize"):
                encoder_loss = l_prior + vae_loss
                self.add_evaluate_metric("encoder_loss", encoder_loss)
                self.encoder_network.optimize_loss(encoder_loss, name="encoder_optimize")
            with tf.name_scope("decoder_optimize"):
                decoder_loss = self.gamma * vae_loss - gan_loss
                self.add_evaluate_metric("decoder_loss", decoder_loss)
                decoder_network.optimize_loss(decoder_loss, name="decoder_optimize")

            # summary images
            self.build_summary_images("decoded", x_tilde, labels=self.get_feed("Y"))
            self.build_summary_images("generated", x_p)

    def optimize(self, batch, feed_map):
        # optimize encoder-network
        encoder_results = self.run("encoder_optimize", feed_map)

        # optimize discriminator-network
        for i in range(self.discriminator_steps):
            discriminator_results = self.run("discriminator_optimize", feed_map)

        # optimize generator-network
        decoder_results = self.run("decoder_optimize", feed_map)

        results = {**encoder_results, **decoder_results, **discriminator_results}

        return results
