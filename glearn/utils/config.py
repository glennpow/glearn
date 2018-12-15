import os
import yaml
import json
import tensorflow as tf
from glearn.datasets import load_dataset
from glearn.envs import load_env
from glearn.utils.log import Loggable
from glearn.utils.path import script_relpath
from glearn.policies.interface import Interface
from glearn.viewers import load_view_controller


TEMP_DIR = "/tmp/glearn"


class Config(object):
    def __init__(self, config_path, version=None, render=False, debug=False):

        self.properties = self.load_properties(config_path)

        # debugging
        self.debugging = debug

        # load env or dataset
        self.seed = self.get("seed", 1)
        tf.set_random_seed(self.seed)
        self.env = None
        self.dataset = None
        if "env" in self.properties:
            # make env
            self.env = load_env(self.properties["env"])
            self.project = self.env.name
        elif "dataset" in self.properties:
            # make dataset
            self.dataset = load_dataset(self.properties)
            self.project = self.dataset.name
        if self.env is None and self.dataset is None:
            raise Exception("Failed to find training env or dataset in config")

        # prepare log and save/load paths
        if version is None:
            next_version = 1
            self.log_dir = f"{TEMP_DIR}/{self.project}/{next_version}"
            self.load_path = None
            self.save_path = f"{self.log_dir}/model.ckpt"
        elif version.isdigit():
            version = int(version)
            next_version = version + 1
            self.log_dir = f"{TEMP_DIR}/{self.project}/{next_version}"
            self.load_path = f"{TEMP_DIR}/{self.project}/{version}/model.ckpt"
            self.save_path = f"{self.log_dir}/model.ckpt"
        else:
            next_version = None
            self.log_dir = f"{TEMP_DIR}/{self.project}/{version}"
            self.load_path = f"{TEMP_DIR}/{self.project}/{version}/model.ckpt"
            self.save_path = None
        self.version = version
        self.tensorboard_path = f"{self.log_dir}/tensorboard/"

        # create render viewer controller
        self.viewer = load_view_controller(self, render=render)

        # prepare input/output interfaces, and env
        if self.supervised:
            self.input = self.dataset.input
            self.output = self.dataset.output
        elif self.reinforcement:
            self.env.seed(self.seed)

            self.input = Interface(self.env.observation_space)
            self.output = Interface(self.env.action_space)

    def load_properties(self, config_path):
        properties = {}

        # load config file
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith("yml"):
                properties = yaml.load(f)
            elif config_path.endswith(".json"):
                properties = json.load(f)

        # include any parent configs  (TODO - could use find_config for these too...)
        if "include" in properties:
            includes = properties["include"]
            if not isinstance(includes, list):
                includes = [includes]
            new_properties = {}
            for include in includes:
                relative_include = os.path.join(os.path.dirname(config_path), include)
                included = self.load_properties(relative_include)
                if included is not None:
                    new_properties.update(included)
            new_properties.update(properties)  # TODO - smart merge
            properties = new_properties

        return properties

    def has(self, key):
        return key in self.properties

    def get(self, key, default=None):
        return self.properties.get(key, default)

    def find(self, key, default=None):
        # find key in all properties
        def _find(properties, key):
            if key in properties:
                return properties[key]
            for _, v in properties.items():
                if isinstance(v, dict):
                    pv = _find(v, key)
                    if pv is not None:
                        return pv
                elif isinstance(v, list):
                    for lv in v:
                        if isinstance(lv, dict):
                            pv = _find(lv, key)
                            if pv is not None:
                                return pv
            return None
        pv = _find(self.properties, key)
        if pv is not None:
            return pv
        return default

    @property
    def reinforcement(self):
        return self.env is not None

    @property
    def supervised(self):
        return self.dataset is not None


class Configurable(Loggable):
    def __init__(self, config):
        self.config = config

    @property
    def debugging(self):
        return self.config.debugging

    @property
    def project(self):
        return self.config.project

    @property
    def dataset(self):
        return self.config.dataset

    @property
    def env(self):
        return self.config.env

    @property
    def supervised(self):
        return self.config.supervised

    @property
    def reinforcement(self):
        return self.config.reinforcement

    @property
    def input(self):
        return self.config.input

    @property
    def output(self):
        return self.config.output

    @property
    def seed(self):
        return self.config.seed


def load_config(identifier, version=None, render=False, debug=False, search_defaults=True):
    # locate the desired config
    config_path = find_config(identifier, search_defaults=search_defaults)
    if config_path is None or not os.path.exists(config_path):
        raise ValueError(f"Failed to locate config using identifier: '{identifier}'")

    return Config(config_path, version=version, render=render, debug=debug)


def find_config(identifier, search_defaults=True):
    # first try the identifier as if is the full path
    options = [identifier]

    if search_defaults:
        # does it need an extension?   (TODO - try more extensions?)
        _, ext = os.path.splitext(identifier)
        if len(ext) == 0:
            options.append(f"{identifier}.yaml")

        # is it relative to the project root?
        root = script_relpath("../..")
        options += [os.path.join(root, "configs", p) for p in options]

    for path in options:
        if os.path.exists(path):
            return path
    return None
