import os
from glearn.policies import load_policy
from glearn.utils.config import load_config


def train(config_path, version=None, render=False, profile=False):
    # load config
    config = load_config(config_path)

    # create policy
    policy = load_policy(config, version=version)

    # train policy
    policy.train(render=render, profile=profile)


def remote_train(config_path):
    # update config path when running on remote machine
    config_path = os.path.join(os.path.dirname(__file__), config_path)

    train(config_path)
