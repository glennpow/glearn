from .network import Network
from .fully_connected import FullyConnectedLayer  # noqa
from .conv2d import Conv2dLayer  # noqa
from .lstm import LSTMLayer  # noqa


def load_network(context, definition):
    network = Network(context, definition)
    network.definition = definition
    return network
