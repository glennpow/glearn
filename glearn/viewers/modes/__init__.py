from glearn.utils.reflection import get_class

from .cnn_viewer_mode import CNNViewerMode  # noqa


def load_viewer_mode(config):
    ViewerModeClass = get_class(config)

    return ViewerModeClass()
