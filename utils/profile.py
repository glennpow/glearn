#!/usr/bin/env python3

import os
import click
from utils.subprocess_utils import shell_call


def open_profile(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            # get latest report
            paths = [f for f in os.listdir(path) if f.startswith("profile_")]
            path = os.path.join(path, sorted(paths)[-1])

        print(f"Opening profiler report: {path}...")
        profiler_ui = "/Users/glennpowell/Workspace/public/profiler-ui/ui.py"
        cmd = ["python", profiler_ui, "--profile_context_path", path]
        shell_call(cmd, verbose=True)


@click.command()
@click.argument("path")
def main(path):
    open_profile(path)


if __name__ == "__main__":
    main()
