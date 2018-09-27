from glearn.utils.reflection import get_class

from .fully_connected import FullyConnectedLayer  # noqa
from .conv2d import Conv2dLayer  # noqa
from .lstm import LSTMLayer  # noqa


def load_layer(index, config):
    LayerClass = get_class(config)

    return LayerClass(index)
