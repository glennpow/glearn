import os


def relative_path(path, relative_to=None):
    if relative_to is None:
        relative_to = os.getcwd()
    else:
        if not os.path.exists(relative_to):
            raise ValueError(f"Relative-to path not found: '{relative_to}'")
        if not os.path.isdir(relative_to):
            # relative to directory of file
            relative_to = os.path.dirname(relative_to)
    # TODO - make both paths are absolute?
    return os.path.join(relative_to, path)
