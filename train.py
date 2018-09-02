#!/usr/bin/env python3

import click
from policies import load_policy
from utils.config import load_config


@click.command()
@click.argument("config_path")
@click.option("--version", "-v", default=None)
@click.option("--render/--no-render", default=False)
@click.option("--profile/--no-profile", default=False)
def main(config_path, version, render, profile):
    # load config
    config = load_config(config_path)

    # create policy
    policy = load_policy(config, version=version)

    # train policy
    policy.train(render=render, profile=profile)


if __name__ == "__main__":
    main()
