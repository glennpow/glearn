import os
import yaml
import json
import gym
from glearn.datasets import load_dataset
from glearn.utils.reflection import get_class
from glearn.policies.interface import Interface
from glearn.viewers import ViewerController


TEMP_DIR = "/tmp/glearn"


class Config(object):
    def __init__(self, config_path, version=None, debug=False):
        self.properties = self.load_properties(config_path)

        # debugging
        self.debugging = debug

        # load env or dataset
        self.seed = self.get("seed", 0)
        self.env = None
        self.dataset = None
        if "env" in self.properties:
            # make env
            env_name = self.properties["env"]
            if isinstance(env_name, dict) or ":" in env_name:
                # use EntryPoint to get env
                EnvClass = get_class(env_name)
                self.env = EnvClass()

                if isinstance(env_name, dict):
                    self.project = env_name['name']
                else:
                    self.project = env_name
                self.project = self.project.split(":")[-1]
            elif "-v" in env_name:
                # use gym to get env
                self.env = gym.make(env_name)
                self.project = env_name
            else:
                raise Exception(f"Unrecognizable environment identifier: {env_name}")
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
            self.save_path = None
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
        self.viewer = ViewerController(self, self.env)

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

    @property
    def reinforcement(self):
        return self.env is not None

    @property
    def supervised(self):
        return self.dataset is not None


def load_config(identifier, version=None, debug=False, search_defaults=True):
    # locate the desired config
    config_path = find_config(identifier, search_defaults=search_defaults)
    if config_path is None or not os.path.exists(config_path):
        raise ValueError(f"Failed to locate config using identifier: '{identifier}'")

    return Config(config_path, version=version, debug=debug)


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
