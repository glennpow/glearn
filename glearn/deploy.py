#!/usr/bin/env python3

import os
import click
from rcall import meta
from glearn.train import train


def remote_train(config_path):
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    train(config_path)


@click.command()
@click.argument("config_path")
@click.option("--backend", "-b", default="local")
def main(config_path, backend):
    job_name = os.path.splitext(os.path.basename(config_path))[0]
    config_path = os.path.relpath(config_path, os.path.dirname(__file__))

    meta.call(
        backend=backend,
        fn=remote_train,
        kwargs=dict(config_path=config_path),
        log_relpath=job_name,
        job_name=job_name,
    )


if __name__ == '__main__':
    main()
