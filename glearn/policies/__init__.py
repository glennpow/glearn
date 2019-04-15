from glearn.utils.reflection import get_class

from .network import NetworkPolicy  # noqa
from .q_network import QNetworkPolicy  # noqa
from .random import RandomPolicy  # noqa


def load_policy(config, context, definition=None):
    if definition is None:
        definition = config.get("policy", None)

    PolicyClass = get_class(definition)

    policy = PolicyClass(config, context)
    policy.definition = definition
    return policy
