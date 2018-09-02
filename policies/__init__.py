from policies.policy_gradient import PolicyGradient
from policies.cnn import CNN
from policies.rnn import RNN


def load_policy(config, version=None):
    name = config.get("policy", None)

    if name == "cnn":
        PolicyClass = CNN
    elif name == "rnn":
        PolicyClass = RNN
    else:
        PolicyClass = PolicyGradient

    return PolicyClass(config, version=version)
