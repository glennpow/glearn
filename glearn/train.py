from glearn.execute import execute


def train(config_path, version=None, render=False, debug=False, profile=False):
    execute(config_path, True, version=version, render=render, debug=debug, profile=profile)
