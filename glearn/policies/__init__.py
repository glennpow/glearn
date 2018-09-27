from glearn.utils.reflection import get_class

from .network import NetworkPolicy  # noqa
from .random import RandomPolicy  # noqa


def load_policy(config, definition=None):
    if definition is None:
        definition = config.get("policy", None)

    PolicyClass = get_class(definition)

    return PolicyClass(config)
