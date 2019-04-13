from glearn.utils.reflection import get_class

from .supervised.supervised import SupervisedTrainer  # noqa
from .reinforcement.policy_gradient import PolicyGradientTrainer  # noqa
from .reinforcement.advantage_actor_critic import AdvantageActorCriticTrainer  # noqa
from .reinforcement.soft_actor_critic import SoftActorCriticTrainer  # noqa
from .reinforcement.proximal_policy_optimization import ProximalPolicyOptimizationTrainer  # noqa
from .reinforcement.deep_q_network import DeepQNetworkTrainer  # noqa
from .unsupervised.variational_autoencoder import VariationalAutoencoderTrainer  # noqa
from .unsupervised.generative_adversarial_network import GenerativeAdversarialNetworkTrainer # noqa
from .unsupervised.vae_gan import VAEGANTrainer  # noqa


def load_trainer(config,):
    definition = config.get("trainer", None)
    if definition is None:
        raise Exception("No Trainer configured!")

    TrainerClass = get_class(definition)

    trainer = TrainerClass(config)
    trainer.definition = definition
    return trainer
