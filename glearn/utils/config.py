import os
import yaml
import json


def load_config(path):
    config = {}

    # load config file
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith("yml"):
            config = yaml.load(f)
        elif path.endswith(".json"):
            config = json.load(f)

    # include any parent configs
    if "include" in config:
        includes = config["include"]
        if not isinstance(includes, list):
            includes = [includes]
        new_config = {}
        for include in includes:
            relative_include = os.path.join(os.path.dirname(path), include)
            included = load_config(relative_include)
            if included is not None:
                new_config.update(included)
        new_config.update(config)
        config = new_config

    return config