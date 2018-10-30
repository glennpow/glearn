def load_view_controller(config, render=True):
    from glearn.viewers.viewer_controller import ViewerController

    return ViewerController(config, render=render)
