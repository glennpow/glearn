from .network import Network
from .layers.dense import DenseLayer  # noqa
from .layers.conv2d import Conv2dLayer  # noqa
from .layers.conv2d_transpose import Conv2dTransposeLayer  # noqa
from .layers.lstm import LSTMLayer  # noqa
from .layers.distributions.normal import NormalDistributionLayer  # noqa
from .layers.distributions.categorical import CategoricalDistributionLayer  # noqa
from .layers.distributions.categorical import DiscretizedDistributionLayer  # noqa


def load_network(name, context, definition, trainable=True):
    network = Network(name, context, definition, trainable=trainable)
    network.definition = definition
    return network
