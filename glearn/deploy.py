#!/usr/bin/env python3

import os
import click
from rcall import meta
from glearn.train import remote_train


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
