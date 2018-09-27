import tensorflow as tf
from glearn.trainers.trainer import Trainer
from glearn.networks import load_network


class ActorCriticTrainer(Trainer):
    def __init__(self, config, policy, critic, **kwargs):
        # get basic params
        self.critic_definition = critic

        super().__init__(config, policy, **kwargs)

    def init_optimizer(self):
        assert hasattr(self.policy, "network")
        policy_definition = self.config.get("policy")
        self.target_network = load_network(self.policy, policy_definition)

        self.critic_network = load_network(self.policy, self.critic_definition)
        self.critic.build
        critic_loss = self.critic.get_fetch("loss", "evaluate")

        # get policy loss
        loss = self.get_loss()

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = self.load_optimizer()
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

    def advantage(self, Q, V):
        # A(s, a) = Q(s, a) - V(s)
        # how good an action is compared to the average action for a state
        return Q - V
