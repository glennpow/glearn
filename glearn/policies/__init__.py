from glearn.utils.reflection import get_class

from glearn.policies.policy_gradient import PolicyGradient  # noqa
from glearn.policies.cnn import CNN  # noqa
from glearn.policies.rnn import RNN  # noqa


def load_policy(config, version=None):
    name = config.get("policy", None)

    PolicyClass = get_class(name)

    return PolicyClass(config, version=version)
