import tensorflow as tf
from .generative import GenerativeTrainer


class GenerativeAdversarialNetworkTrainer(GenerativeTrainer):
    def __init__(self, config, discriminator=None, generator=None, discriminator_steps=1,
                 discriminator_scale_factor=None, generator_scale_factor=None,
                 alternative_generator_loss=False, summary_images=9, fixed_evaluate_noise=False,
                 **kwargs):
        # get basic params
        self.discriminator_definition = discriminator
        self.discriminator_steps = discriminator_steps
        self.discriminator_scale_factor = discriminator_scale_factor
        self.generator_definition = generator
        self.generator_scale_factor = generator_scale_factor
        self.alternative_generator_loss = alternative_generator_loss
        self.summary_images = summary_images

        self.discriminator_networks = []

        super().__init__(config, **kwargs)

    def build_discriminator(self, x, real):
        # build network
        definition = self.discriminator_definition
        reuse = len(self.discriminator_networks) > 0
        network = self.build_network("discriminator", definition, x, reuse=reuse)

        # set labels and logits
        network.logits = network.get_output_layer().references["Z"]
        network.real = real
        network.scale_factor = self.discriminator_scale_factor if real else None

        # append network
        self.discriminator_networks.append(network)
        return network.outputs

    def build_generator(self):
        definition = self.generator_definition
        self.generator_network = self.build_generator_network("generator", definition)
        return self.generator_network.outputs

    def build_discriminator_optimize(self, loss_name="discriminator_loss"):
        with self.variable_scope("discriminator_optimize"):
            # sum losses from all discriminator networks
            loss = 0
            for network in self.discriminator_networks:
                # prepare labels
                if network.real:
                    labels = tf.ones_like(network.logits)
                else:
                    labels = tf.zeros_like(network.logits)
                if network.scale_factor is not None:
                    labels -= network.scale_factor

                # cross entropy loss
                network.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=network.logits,
                                                                       labels=labels)
                loss += tf.reduce_mean(network.loss)

            # loss summary
            if loss_name:
                self.add_metric(loss_name, loss)

            # optimize loss for networks
            self.discriminator_networks[0].optimize_loss(loss, name="discriminator_optimize")
        return loss

    def build_generator_optimize(self, loss_name="generator_loss"):
        with self.variable_scope("generator_optimize"):
            # sum losses from all fake discriminator networks
            loss = 0
            for network in self.discriminator_networks:
                if network.real:
                    continue

                if self.alternative_generator_loss:
                    # alternative of using negative discriminator loss  (FIXME)
                    network_loss = -network.loss
                else:
                    # prepare negative labels
                    labels = tf.ones_like(network.logits)
                    if self.generator_scale_factor is not None:
                        labels -= self.generator_scale_factor

                    # cross entropy loss of negative labels
                    network_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=network.logits,
                                                                           labels=labels)
                loss += tf.reduce_mean(network_loss)

            # loss summary
            if loss_name:
                self.add_metric(loss_name, loss)

            # optimize loss for network
            self.generator_network.optimize_loss(loss, name="generator_optimize")
        return loss

    def build_trainer(self):
        super().build_trainer()

        with self.variable_scope("GAN"):
            # build generator-network
            generated = self.build_generator()

            # normalize real input
            with self.variable_scope("normalize_images"):
                x = self.get_feed("X")
                x = x * 2 - 1  # normalize (-1, 1)

            # build discriminator-networks
            self.build_discriminator(x, True)
            self.build_discriminator(generated, False)

            # optimize discriminator loss
            self.build_discriminator_optimize()

            # optimize generator loss
            self.build_generator_optimize()

            # summary images
            self.build_summary_images("generated", generated)

    def optimize(self, batch):
        feed_map = batch.get_feeds()

        # optimize discriminator-network
        for i in range(self.discriminator_steps):
            self.run("discriminator_optimize", feed_map)

        # optimize generator-network
        results = self.run("generator_optimize", feed_map)

        return results
