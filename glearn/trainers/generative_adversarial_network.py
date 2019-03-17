import tensorflow as tf
from .generative import GenerativeTrainer


class GenerativeAdversarialNetworkTrainer(GenerativeTrainer):
    def __init__(self, config, discriminator=None, generator=None, discriminator_steps=1,
                 discriminator_scale_factor=None, generator_scale_factor=None,
                 alternative_generator_loss=True, summary_images=4, fixed_evaluate_noise=False,
                 **kwargs):
        # get basic params
        self.discriminator_definition = discriminator
        self.discriminator_steps = discriminator_steps
        self.discriminator_scale_factor = discriminator_scale_factor
        self.generator_definition = generator
        self.generator_scale_factor = generator_scale_factor
        self.alternative_generator_loss = alternative_generator_loss
        self.summary_images = summary_images
        self.fixed_evaluate_noise = fixed_evaluate_noise

        self.discriminator_networks = []

        super().__init__(config, **kwargs)

        # only works for datasets
        assert(self.has_dataset)

    def build_discriminator(self, x, real, scale_factor=None):
        # build network
        definition = self.discriminator_definition
        reuse = len(self.discriminator_networks) > 0
        network = self.build_network("discriminator", definition, x, reuse=reuse)

        # set labels and logits
        network.logits = network.get_output_layer().references["Z"]
        network.real = real
        network.scale_factor = scale_factor

        # append network
        self.discriminator_networks.append(network)
        return network.outputs

    def build_generator(self):
        definition = self.generator_definition
        self.generator_network = self.build_generator_network("generator", definition)
        return self.generator_network.outputs

    def build_discriminator_loss(self, loss_name="discriminator_loss"):
        with tf.variable_scope("discriminator_optimize"):
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
                self.add_evaluate_metric(loss_name, loss)

            # optimize loss for networks
            self.discriminator_networks[0].optimize_loss(loss, name="discriminator_optimize")
        return loss

    def build_generator_loss(self, loss_name="generator_loss", scale_factor=None):
        with tf.variable_scope("generator_optimize"):
            # sum losses from all fake discriminator networks
            loss = 0
            for network in self.discriminator_networks:
                if network.real:
                    continue

                if self.alternative_generator_loss:
                    # alternative of using negative discriminator loss
                    network_loss = -network.loss
                else:
                    # prepare negative labels
                    labels = tf.ones_like(network.logits)
                    if scale_factor is not None:
                        labels -= scale_factor

                    # cross entropy loss of negative labels
                    network_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=network.logits,
                                                                           labels=labels)
                loss += tf.reduce_mean(network_loss)

            # loss summary
            if loss_name:
                self.add_evaluate_metric(loss_name, loss)

            # optimize loss for network
            self.generator_network.optimize_loss(loss, name="generator_optimize")
        return loss

    def build_gan_summary_images(self, generated):
        # generated image summaries
        with tf.variable_scope("summary_images"):
            images = tf.reshape(generated, tf.shape(self.get_feed("X")))
            self.summary.add_images(f"generated", images, self.summary_images)

    def build_trainer(self):
        # evaluate can use fixed noise
        if self.fixed_evaluate_noise:
            epoch_size = self.dataset.get_epoch_size("test")
            self.evaluate_latents = [self.sample_noise(self.batch_size) for i in range(epoch_size)]

        with tf.variable_scope("gan"):
            # build generator-network
            generated = self.build_generator()

            # normalize real input
            with tf.name_scope("normalize_images"):
                x = self.get_feed("X")
                x = x * 2 - 1  # normalize (-1, 1)

            # build discriminator-networks
            self.build_discriminator(x, True, scale_factor=self.discriminator_scale_factor)
            self.build_discriminator(generated, False)

            # optimize discriminator loss
            self.build_discriminator_loss()

            # optimize generator loss
            self.build_generator_loss(scale_factor=self.generator_scale_factor)

            # summary images
            self.build_gan_summary_images(generated)

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
            latent = self.sample_noise(self.batch_size)
        feed_map["Z"] = latent

        return super().prepare_feeds(queries, feed_map)

    def optimize(self, batch, feed_map):
        # optimize discriminator-network
        for i in range(self.discriminator_steps):
            self.run("discriminator_optimize", feed_map)

        # optimize generator-network
        results = self.run("generator_optimize", feed_map)

        return results
