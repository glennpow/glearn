import os
import inspect


def script_relpath(relative_path):
    # return an absolute path constructed relative to the calling script (TODO - or module)
    start = os.path.dirname(inspect.stack()[1][1])
    return os.path.abspath(os.path.join(start, relative_path))
