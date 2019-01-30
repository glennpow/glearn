from glearn.train import train


def evaluate(config_path, version=None, render=False, debug=False):
    train(config_path, version=version, render=render, debug=debug, training=False)
