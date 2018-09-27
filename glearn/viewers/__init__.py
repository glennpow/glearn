from glearn.viewers.advanced_viewer import AdvancedViewer  # noqa


def load_view_controller(config):
    from glearn.viewers.viewer_controller import ViewerController
    return ViewerController(config)
