import numpy as np
import tensorflow as tf
from glearn.trainers.trainer import Trainer


class PolicyGradientTrainer(Trainer):
    def __init__(self, config, policy,
                 optimizer="sgd", learning_rate=1e-3, epsilon=0, keep_prob=1, max_grad_norm=None,
                 **kwargs):
        # get basic params
        self.optimizer = optimizer
        self.learning_rate = learning_rate  # lambda λ
        self.epsilon = epsilon
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm

        super().__init__(config, policy, **kwargs)

    def init_optimizer(self):
        # get loss from policy
        loss = self.policy.get_fetch("loss", "evaluate")
        if loss is None:
            self.error(f"Policy ({self.policy}) does not define a 'loss' feed for 'evaluate'")
            return
        self.summary.add_scalar("loss", loss, "evaluate")

        # TODO - loss *= discounted_rewards (gamma)

        # minimize loss
        with tf.name_scope('optimize'):
            # create optimizer
            if self.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

                # self.policy.set_fetch("learning_rate", optimizer._learning_rate, "debug")
            elif self.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                # TODO - this is only populated later, for some reason
                # self.policy.set_fetch("learning_rate", optimizer._lr_t, "debug")
            else:
                raise Exception(f"Unknown optimizer type specified in config: {self.optimizer}")
            global_step = tf.train.get_or_create_global_step()

            # apply gradients, with any configured clipping
            max_grad_norm = self.max_grad_norm
            if max_grad_norm is None:
                # apply unclipped gradients
                optimize = optimizer.minimize(loss, global_step=global_step)
            else:
                # apply gradients with clipping
                tvars = tf.trainable_variables()
                grads = tf.gradients(loss, tvars)
                grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
                optimize = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            self.policy.set_fetch("optimize", optimize, "optimize")

    def prepare_feeds(self, graphs, feed_map):
        feed_map = super().prepare_feeds(graphs, feed_map)

        # if graph == "optimize" or graph == "evaluate":
        #     # should this be per epoch/episode instead of iteration?
        #     max_lr_step = 10
        #     lr_decay = self.lr_decay ** max(self.global_step + 1 - max_lr_step, 0.0)
        #     learning_rate = self.learning_rate * lr_decay
        #     feed_map["lambda"] = learning_rate

        if "optimize" in graphs:
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1
        return feed_map

    def action(self):
        # decaying epsilon-greedy
        # FIXME - should this be per epoch/episode instead of iteration?
        epsilon = self.epsilon
        if isinstance(epsilon, list):
            t = min(1, self.global_step / epsilon[2])
            epsilon = t * (epsilon[1] - epsilon[0]) + epsilon[0]

        # get action
        if np.random.random() < epsilon:
            # choose epsilon-greedy random action  (TODO - could implement this in tf)
            return self.output.sample()
        else:
            # choose optimal policy action
            return super().action()
