import tensorflow as tf
from .variational_autoencoder import VariationalAutoencoderTrainer
from .generative_adversarial_network import GenerativeAdversarialNetworkTrainer


class VAEGANTrainer(VariationalAutoencoderTrainer, GenerativeAdversarialNetworkTrainer):
    def __init__(self, config, **kwargs):
        # TODO - parse extra config params

        VariationalAutoencoderTrainer.__init__(self, config, **kwargs)
        GenerativeAdversarialNetworkTrainer.__init__(self, config, **kwargs)

    def build_trainer(self):
        with tf.variable_scope("vae_gan"):
            # real input
            x = self.get_feed("X")

            # build encoder-network
            encoded = self.build_encoder(x)

            # build decoder-network
            x_hat = self.build_decoder(encoded)

            # build VAE loss
            vae_loss = self.build_vae_loss(x, x_hat, optimize_decoder=False, name="vae_loss")

            # build generator-network
            x_p = self.build_decoder(reuse=True)
            x_p = x_p * 2 - 1  # normalize (-1, 1)  # HACK?

            # build discriminator-networks
            self.real_discriminator_network = self.build_discriminator(x)
            self.fake_discriminator_network = self.build_discriminator(x_p, reuse=True)
            self.fake_discriminator_network = self.build_discriminator(x_p, reuse=True)
            self.build_discriminator(x, x_p)

            # optimize discriminator loss
            self.build_discriminator_loss()

            # TODO - need to also optimize discriminator on decoded images?

            # optimize generator loss
            l_prior = self.get_fetch("kl_divergence")
            gan_loss = self.build_generator_loss(optimize=False)

            # summary images
            self.build_vae_summary_images(x, decoded)
            self.build_gan_summary_images(generated)

        # minimize generator loss  TODO - use generic optimize_loss
        with tf.variable_scope("vae_gan_optimize"):
            loss = vae_loss + gan_loss  # FIXME - some linear combination?

            optimize_networks = [self.generator_network]
            self.optimize_loss(loss, networks=optimize_networks, name="generator_optimize")

    def optimize(self, batch, feed_map):
        # optimize encoder-network
        vae_results = self.run("encoder_optimize", feed_map)

        # optimize discriminator-network
        for i in range(self.discriminator_steps):
            discriminator_results = self.run("discriminator_optimize", feed_map)

        # optimize generator-network
        generator_results = self.run("generator_optimize", feed_map)

        results = {**vae_results, **generator_results, **discriminator_results}

        return results
