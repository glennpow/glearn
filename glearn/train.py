#!/usr/bin/env python3

import os
import click
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
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    train(config_path)


@click.command()
@click.argument("config_path")
@click.option("--version", "-v", default=None)
@click.option("--render/--no-render", default=False)
@click.option("--profile/--no-profile", default=False)
def main(config_path, version, render, profile):
    train(config_path, version=version, render=render, profile=profile)


if __name__ == "__main__":
    main()
