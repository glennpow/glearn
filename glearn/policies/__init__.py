from glearn.utils.reflection import get_class

from .network import NetworkPolicy  # noqa
from .random import RandomPolicy  # noqa


def load_policy(config):
    name = config.get("policy", None)

    PolicyClass = get_class(name)

    return PolicyClass(config)
