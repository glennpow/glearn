import os
from glearn.policies import load_policy
from glearn.trainers import load_trainer
from glearn.utils.config import load_config


def train(config_path, version=None, render=False, debug=False, profile=False):
    # load config
    config = load_config(config_path, version=version, debug=debug)

    # create policy
    policy = load_policy(config)

    # create trainer
    trainer = load_trainer(config, policy)

    # train policy
    trainer.train(render=render, profile=profile)


def remote_train(config_path, **kwargs):
    # update config path when running on remote machine
    config_path = os.path.join(os.path.dirname(__file__), config_path)

    train(config_path, **kwargs)
