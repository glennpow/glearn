#!/usr/bin/env python3

import os
import click
import tensorflow as tf
from glearn.utils.subprocess_utils import shell_call


def run_profile(call, sess, config):
    # get profile options
    path = f"{config.log_dir}/profile"
    profile_config = config.get("profile", None)
    profile_options = None
    if profile_config is not None:
        if "time_and_memory" in profile_config:
            profile_options = tf.profile.ProfileOptionsBuilder.time_and_memory()

    # run call with profiler
    with tf.contrib.tfprof.ProfileContext(path) as pctx:
        pctx.profiler.profile_operations(run_meta=sess.run_metadata, options=profile_options)
        call()

    return path


def open_profile(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            # get latest report
            paths = [f for f in os.listdir(path) if f.startswith("profile_")]
            path = os.path.join(path, sorted(paths)[-1])

        print(f"Opening profiler report: {path}...")
        profiler_ui = "/Users/glennpowell/Workspace/public/profiler-ui/ui.py"  # TODO
        cmd = ["python", profiler_ui, "--profile_context_path", path]
        shell_call(cmd, verbose=True)


@click.command()
@click.argument("path")
def main(path):
    open_profile(path)


if __name__ == "__main__":
    main()
