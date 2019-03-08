import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.networks import load_network
from glearn.datasets.labeled import LabeledDataset


class VariationalAutoencoderTrainer(Trainer):
    def __init__(self, config, decoder, **kwargs):
        # get basic params
        self.decoder_definition = decoder

        super().__init__(config, **kwargs)

        # only works for labeled datasets
        assert(self.has_dataset)
        assert(isinstance(self.dataset, LabeledDataset))

    def learning_type(self):
        return "unsupervised"

    def build_trainer(self):
        with tf.variable_scope("vae"):
            # get original images
            x = self.get_feed("X")
            x_shape = tf.shape(x)
            x_size = tf.reduce_prod(x_shape[1:])
            x = tf.reshape(x, [-1, x_size])

            # build decoder-network
            decoder_input = self.get_fetch("predict")
            self.decoder_network = load_network(f"decoder", self, self.decoder_definition)
            y = self.decoder_network.build_predict(decoder_input)
            y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
            self.add_fetch("decoder", y)

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

    def optimize(self, batch, feed_map):
        # optimize encoder/decoder-networks
        results = self.run("vae_update", feed_map)

        return results
