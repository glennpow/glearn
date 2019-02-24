from glearn.execute import execute


def evaluate(config_path, version=None, render=False, debug=False, random=False):
    execute(config_path, False, version=version, render=render, debug=debug, random=random)
