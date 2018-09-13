from glearn.utils.reflection import get_class

from glearn.policies.policy_gradient import PolicyGradientPolicy  # noqa
from glearn.policies.cnn import CNNPolicy  # noqa
from glearn.policies.rnn import RNNPolicy  # noqa
from glearn.policies.random import RandomPolicy  # noqa


def load_policy(config, version=None):
    name = config.get("policy", None)

    PolicyClass = get_class(name)

    return PolicyClass(config, version=version)
