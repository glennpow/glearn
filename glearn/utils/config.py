import os
import yaml
import json


def load_config(identifier, search_defaults=True):
    config = {}

    # locate the desired config
    config_path = find_config(identifier, search_defaults=search_defaults)
    if config_path is None or not os.path.exists(config_path):
        raise ValueError(f"Failed to locate config using identifier: '{identifier}'")

    # load config file
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith("yml"):
            config = yaml.load(f)
        elif config_path.endswith(".json"):
            config = json.load(f)

    # include any parent configs  (TODO - could use find_config for these too...)
    if "include" in config:
        includes = config["include"]
        if not isinstance(includes, list):
            includes = [includes]
        new_config = {}
        for include in includes:
            relative_include = os.path.join(os.path.dirname(config_path), include)
            included = load_config(relative_include)
            if included is not None:
                new_config.update(included)
        new_config.update(config)
        config = new_config

    return config


def find_config(identifier, search_defaults=True):
    # first try the identifier as if is the full path
    options = [identifier]

    if search_defaults:
        # does it need an extension?   (TODO - try more extensions?)
        _, ext = os.path.splitext(identifier)
        if len(ext) == 0:
            options.append(f"{identifier}.yaml")

        # is it relative to the project root?
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        options += [os.path.join(root, "configs", "experiments", p) for p in options]

    for path in options:
        if os.path.exists(path):
            return path
    return None
