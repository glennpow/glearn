from glearn.execute import execute


def evaluate(config_path, version=None, render=False, debug=False):
    execute(config_path, version=version, render=render, debug=debug, training=False)
