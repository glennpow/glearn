import os
import yaml
import json
import collections
import copy
import itertools
import socket
import tensorflow as tf
from glearn.datasets import load_dataset
from glearn.envs import load_env
from glearn.utils.log import log, log_warning, Loggable
from glearn.utils.printing import print_tabular
from glearn.utils.path import script_relpath
from glearn.utils.session import DebuggableSession
from glearn.utils.summary import SummaryWriter, NullSummaryWriter
from glearn.utils.subprocess_utils import shell_call
from glearn.policies.interface import Interface
from glearn.viewers import load_view_controller


TEMP_DIR = "/tmp/glearn"


class Config(object):
    def __init__(self, path, version=None, render=False, debug=False):
        self.path = path
        self.version = version or 1
        self.rendering = render
        self.debugging = debug

        # determine local or external IP address
        local_ip = socket.gethostbyname(socket.gethostname())
        self.local = local_ip == "127.0.0.1"
        if self.local:
            self.ip = local_ip
        else:
            self.ip = shell_call(["dig", "+short", "myip.opendns.com", "@resolver1.opendns.com"],
                                 response_type="text", ignore_exceptions=True)

        # load properties from config file
        self.properties = self._load_properties(self.path, local=self.local)

        # debugging
        if self.debugging:
            from glearn.utils.debug import debug_faults
            debug_faults()

        self.sess = None
        self.summary = None
        self._load_evaluations()

    def _load_properties(self, config_path, local=False):
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
                import_properties = self._load_properties(import_path)
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

    def __iter__(self):
        # prepare to iterate through evaluations
        self.current_evaluation = 0
        return self

    def __next__(self):
        # iterate through evaluations
        self.num_evaluations = len(self.evaluations) if self.evaluations else 1
        if self.current_evaluation < self.num_evaluations:
            self._start_evaluation()
            self.current_evaluation += 1
            return self
        else:
            raise StopIteration

    def _load_evaluations(self):
        self.evaluations = None
        self.current_evaluation = 0
        self.current_sweep = None

        # combine sweeps into evaluations
        sweeps = self.properties.pop("sweeps", None)
        if sweeps:
            def _build_combinations(d):
                params = list(d.keys())
                values = list(d.values())
                combinations = itertools.product(*values)
                return [list(zip(params, combination)) for combination in combinations]

            # build sweeps evaluations from dict or list of dicts
            if isinstance(sweeps, list):
                self.evaluations = [c for d in sweeps for c in _build_combinations(d)]
            elif isinstance(sweeps, dict):
                self.evaluations = _build_combinations(sweeps)

    def _start_evaluation(self):
        message = f"Starting Evaluation: {self.current_evaluation + 1} / {self.num_evaluations}"

        if self.evaluations:
            # copy params for this evaluation
            self.current_properties = copy.deepcopy(self.properties)
            evaluation_sweep = self.evaluations[self.current_evaluation]
            self.current_sweep = dict(evaluation_sweep)

            def _recursive_replace(node, param, value, path=None):
                num_replaced_values = 0
                if isinstance(node, list):
                    for i in range(len(node)):
                        v_path = f"{path}[{i}]"
                        num_replaced_values += _recursive_replace(node[i], param, value, v_path)
                elif isinstance(node, dict):
                    for k, v in node.items():
                        v_path = k if not path else "{}.{}".format(path, k)
                        if v_path.endswith(param):
                            node[k] = value
                            num_replaced_values += 1
                        else:
                            num_replaced_values += _recursive_replace(v, param, value, v_path)
                return num_replaced_values

            # set appropriate hyperparams for this sweep evaluation
            for param, value in evaluation_sweep:
                replacements = _recursive_replace(self.current_properties, param, value)
                if replacements == 0:
                    log_warning(f"Sweeps parameter '{param}' not found in config.")

            # log evaluation sweep info
            print_tabular({message: self.current_sweep}, grouped=True, color="white", bold=True)
        else:
            self.current_properties = self.properties

            # log evaluation info
            log(message, color="white", bold=True)

        # init session
        self._init_session()

        # load env or dataset
        self.seed = self.get("seed", 1)
        tf.set_random_seed(self.seed)
        self.env = None
        self.dataset = None
        if self.has("env"):
            # make env
            self.env = load_env(self.get("env"))
            self.project = self.env.name
        elif self.has("dataset"):
            # make dataset
            self.dataset = load_dataset(self)
            self.project = self.dataset.name
        if self.env is None and self.dataset is None:
            raise Exception("Failed to find training env or dataset in config")

        # prepare log and save/load paths
        self.log_dir = f"{TEMP_DIR}/{self.project}/{self.version}"
        self.summary_path = f"{self.log_dir}/summaries/"
        self.tensorboard_path = self.summary_path
        if self.evaluations:
            self.log_dir = f"{self.log_dir}/{self.current_evaluation + 1}"
            self.summary_path = f"{self.summary_path}/{self.current_evaluation + 1}"
        self.load_path = f"{self.log_dir}/model.ckpt"
        self.save_path = self.load_path

        # create render viewer controller
        self.viewer = load_view_controller(self, render=self.rendering)

        # prepare input/output interfaces, and env
        if self.supervised:
            self.input = self.dataset.input
            self.output = self.dataset.output
        elif self.reinforcement:
            self.env.seed(self.seed)

            self.input = Interface(self.env.observation_space)
            # FIXME - network output should determine if stochastic (distribution) or deterministic
            self.output = Interface(self.env.action_space, deterministic=False)

        # start summary logging and tensorboard
        self._start_summaries()

    def has(self, key):
        return key in self.current_properties

    def get(self, key, default=None):
        return self.current_properties.get(key, default)

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
        pv = _find(self.current_properties, key)
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
        # set default session
        self.sess = DebuggableSession(self)

    def start_session(self):
        # initialize all global variables
        self.sess.run(tf.global_variables_initializer())

    def stop_session(self):
        if self.sess:
            # stop summary logging and tensorboard
            final_evaluation = self.current_evaluation + 1 >= self.num_evaluations
            self._stop_summaries(stop_server=final_evaluation)

            self.sess.close()
            self.sess = None

            # reset default graph
            tf.reset_default_graph()

            if self.env:
                self.env.close()

    def _start_summaries(self):
        if self.summary is None:
            if self.summary_path is not None:
                self.summary = SummaryWriter(self)
            else:
                self.summary = NullSummaryWriter()
        self.summary.start()

    def _stop_summaries(self, stop_server=True):
        if self.summary is not None:
            self.summary.stop(stop_server=stop_server)


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
