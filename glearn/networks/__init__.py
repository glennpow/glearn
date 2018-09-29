from .network import Network
from .dense import DenseLayer  # noqa
from .conv2d import Conv2dLayer  # noqa
from .lstm import LSTMLayer  # noqa
from .distribution import DistributionLayer  # noqa


def load_network(name, context, definition, trainable=True):
    network = Network(name, context, definition, trainable=trainable)
    network.definition = definition
    return network
