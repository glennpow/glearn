from glearn.utils.reflection import get_class

from .trainer import Trainer
from .policy_gradient import PolicyGradientTrainer  # noqa
from .advantage_actor_critic import AdvantageActorCriticTrainer  # noqa
from .soft_actor_critic import SoftActorCriticTrainer  # noqa
from .ppo import PPOTrainer  # noqa
from .variational_autoencoder import VariationalAutoencoderTrainer  # noqa
from .generative_adversarial_network import GenerativeAdversarialNetworkTrainer  # noqa
from .vae_gan import VAEGANTrainer  # noqa


def load_trainer(config,):
    definition = config.get("trainer", None)
    if definition is None:
        print("No trainer configured, using default.")
        return Trainer(config)

    TrainerClass = get_class(definition)

    trainer = TrainerClass(config)
    trainer.definition = definition
    return trainer
