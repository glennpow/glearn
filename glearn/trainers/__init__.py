from glearn.utils.reflection import get_class

from .trainer import Trainer
from .policy_gradient import PolicyGradientTrainer  # noqa
from .advantage_actor_critic import AdvantageActorCriticTrainer  # noqa
from .soft_actor_critic import SoftActorCriticTrainer  # noqa
from .ppo import PPOTrainer  # noqa
from .vae import VariationalAutoencoderTrainer  # noqa


def load_trainer(config, policy):
    definition = config.get("trainer", None)
    if definition is None:
        print("No trainer configured, using default.")
        return Trainer(config, policy)

    TrainerClass = get_class(definition)

    trainer = TrainerClass(config, policy)
    trainer.definition = definition
    return trainer
