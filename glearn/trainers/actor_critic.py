import tensorflow as tf
from glearn.trainers.trainer import Trainer


class ActorCriticTrainer(Trainer):
    def __init__(self, config, policy):
        # get basic params
        self.learning_rate = config.get("learning_rate", 1e-3)  # lambda λ
        self.keep_prob = config.get("keep_prob", 1)

        super().__init__(config, policy)

    def init_optimizer(self):
        # # get loss from policy
        # loss = self.policy.get_fetch("loss", "evaluate")
        # if loss is None:
        #     raise Exception(f"Policy ({self.policy}) does not define a 'loss' feed for 'evaluate'")
        # self.summary.add_scalar("loss", loss, "evaluate")

        # # TODO - loss *= discounted_rewards

        # # minimize loss
        # with tf.name_scope('optimize'):
        #     # create optimizer
        #     optimizer_name = self.config.get("optimizer", "sgd")
        #     if optimizer_name == "sgd":
        #         optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        #         # self.policy.set_fetch("learning_rate", optimizer._learning_rate, "debug")
        #     elif optimizer_name == "adam":
        #         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        #         # TODO - this is only populated later, for some reason
        #         # self.policy.set_fetch("learning_rate", optimizer._lr_t, "debug")
        #     else:
        #         raise Exception(f"Unknown optimizer type specified in config: {optimizer_name}")
        #     global_step = tf.train.get_or_create_global_step()

        #     # apply gradients, with any configured clipping
        #     max_grad_norm = self.config.get("max_grad_norm", None)
        #     if max_grad_norm is None:
        #         # apply unclipped gradients
        #         optimize = optimizer.minimize(loss, global_step=global_step)
        #     else:
        #         # apply gradients with clipping
        #         tvars = tf.trainable_variables()
        #         grads = tf.gradients(loss, tvars)
        #         grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        #         optimize = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        #     self.policy.set_fetch("optimize", optimize, "optimize")
