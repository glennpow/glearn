import os
import inspect


TEMP_DIR = "/tmp/glearn"


def script_relpath(relative_path):
    # return an absolute path constructed relative to the calling script (TODO - or module)
    start = os.path.dirname(inspect.stack()[1][1])
    return os.path.abspath(os.path.join(start, relative_path))


def remove_empty_dirs(path, remove_root=True):
    if not os.path.isdir(path):
        return

    # remove empty subdirs
    children = os.listdir(path)
    if len(children):
        for child in children:
            child_path = os.path.join(path, child)
            if os.path.isdir(child_path):
                remove_empty_dirs(child_path)

    # if dir is empty, delete it
    children = os.listdir(path)
    if len(children) == 0 and remove_root:
        os.rmdir(path)
