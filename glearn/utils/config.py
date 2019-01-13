import os
import yaml
import json
import collections
import socket
import tensorflow as tf
from glearn.datasets import load_dataset
from glearn.envs import load_env
from glearn.utils.log import Loggable
from glearn.utils.path import script_relpath
from glearn.utils.session import DebuggableSession
from glearn.utils.summary import SummaryWriter, NullSummaryWriter
from glearn.utils.subprocess_utils import shell_call
from glearn.policies.interface import Interface
from glearn.viewers import load_view_controller


TEMP_DIR = "/tmp/glearn"


class Config(object):
    def __init__(self, config_path, version=None, render=False, debug=False):
        # determine local or external IP address
        local_ip = socket.gethostbyname(socket.gethostname())
        self.local = local_ip == "127.0.0.1"
        if self.local:
            self.ip = local_ip
        else:
            self.ip = shell_call(["dig", "+short", "myip.opendns.com", "@resolver1.opendns.com"],
                                 response_type="text", ignore_exceptions=True)

        # load properties from config file
        self.properties = self.load_properties(config_path, local=self.local)

        # debugging
        self.debugging = debug
        if self.debugging:
            from glearn.utils.debug import debug_faults
            debug_faults()

        # init session
        self._init_session()

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
            self.dataset = load_dataset(self)
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

        # init tensorboard summaries and server
        self._init_summaries()

    def load_properties(self, config_path, local=False):
        # load main config file
        properties = None
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith("yml"):
                properties = yaml.load(f)
            elif config_path.endswith(".json"):
                properties = json.load(f)
        if properties is None:
            return {}

        # load imported config files
        imports = properties.pop("import", None)
        if imports is not None:
            if not isinstance(imports, list):
                imports = [imports]

            new_properties = {}
            for import_identifier in imports:
                import_path = find_config(import_identifier)
                import_properties = self.load_properties(import_path)
                if import_properties is not None:
                    self._deep_update(new_properties, import_properties)

            properties = self._deep_update(new_properties, properties)

        # apply any local-specific properties
        local_properties = properties.pop("local", None)
        if local and local_properties is not None:
            self._deep_update(properties, local_properties)

        return properties

    def _deep_update(self, target, src):
        for k, v in src.items():
            if isinstance(v, collections.Mapping):
                target[k] = self._deep_update(target.get(k, {}), v)
            else:
                target[k] = v
        return target

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

    def is_debugging(self, key):
        return self.debugging and self.get(key, False)

    @property
    def reinforcement(self):
        return self.env is not None

    @property
    def supervised(self):
        return self.dataset is not None

    def _init_session(self):
        self.sess = DebuggableSession(self)

    def start_session(self):
        self.sess.run(tf.global_variables_initializer())

        self._start_summaries()

    def stop_session(self):
        self._stop_summaries()

        self.sess.close()

    def _init_summaries(self):
        if self.tensorboard_path is not None:
            self.summary = SummaryWriter(self)
        else:
            self.summary = NullSummaryWriter()

    def _start_summaries(self):
        if self.summary is not None:
            self.summary.start()

    def _stop_summaries(self):
        if self.summary is not None:
            self.summary.stop()


class Configurable(Loggable):
    def __init__(self, config):
        self.config = config

    @property
    def debugging(self):
        return self.config.debugging

    @property
    def is_debugging(self, key):
        return self.condfig.is_debugging(key)

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

    @property
    def sess(self):
        return self.config.sess

    @property
    def summary(self):
        return self.config.summary


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
        # does it need an extension?
        _, ext = os.path.splitext(identifier)
        if len(ext) == 0:
            options.append(f"{identifier}.yaml")
            options.append(f"{identifier}.json")

        # is it relative to the project root?
        root = script_relpath("../..")
        options += [os.path.join(root, "configs", p) for p in options]

    for path in options:
        if os.path.exists(path):
            return path
    return None
