from glearn.utils.reflection import get_class

from .cnn_viewer_mode import CNNViewerMode  # noqa
from .rnn_viewer_mode import RNNViewerMode  # noqa


def load_viewer_mode(config):
    ViewerModeClass = get_class(config)

    return ViewerModeClass()
