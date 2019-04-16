import os
import time
import yaml
import json
import collections
import copy
import itertools
import socket
import numpy as np
import tensorflow as tf
from glearn.datasets import load_dataset
from glearn.envs import load_env
from glearn.utils.log import log, log_warning, log_error, Loggable
from glearn.utils.path import TEMP_DIR
from glearn.utils.printing import print_tabular
from glearn.utils.path import script_relpath
from glearn.utils.session import DebuggableSession
from glearn.utils.summary import SummaryWriter, NullSummaryWriter
from glearn.utils.subprocess_utils import shell_call
from glearn.data.interface import Interface
from glearn.viewers import load_view_controller


CONFIG_EXTENSIONS = [".yaml", ".yml", "json"]


class Config(object):
    def __init__(self, path, version=None, render=False, debug=False, training=False):
        self.path = path
        self.version = version or 1
        self.rendering = render
        self.debugging = debug
        self.training = training

        # get default log paths
        config_name, _ = os.path.splitext(os.path.basename(path))
        self.root_log_dir = f"{TEMP_DIR}/experiments/{config_name}/{self.version}"
        self.tensorboard_path = f"{self.root_log_dir}"

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

        # catch segfaults when debugging
        if self.debugging:
            from glearn.utils.debug import debug_faults
            debug_faults()

        self.sess = None
        self.summary = None
        self.loading = not training or version is not None
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

    def __getitem__(self, indices):
        assert isinstance(indices, int)
        self.current_evaluation = indices
        if self.current_evaluation < self.num_evaluations:
            self._start_evaluation()
            return self
        else:
            raise IndexError

    def __iter__(self):
        # prepare to iterate through evaluations
        self.current_evaluation = 0
        return self

    def __next__(self):
        # iterate through evaluations
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

        # get evaluation generation params
        sweeps = self.properties.pop("sweeps", None)
        seeds = self.properties.pop("seeds", None)

        # allow multiple evaluations with different seeds
        if seeds:
            try:
                if sweeps is None:
                    # default to be overridden
                    self.properties["seed"] = 1

                    if isinstance(seeds, list):
                        self.evaluations = [[["seed", seed]] for seed in seeds]
                    else:
                        seeds = int(seeds)

                        M = 0xffffffff
                        self.evaluations = [[["seed", np.random.randint(M)]] for _ in range(seeds)]
                        self.properties["seed"] = 1
                else:
                    log_warning("Currently 'seeds' is ignored when 'sweeps' config is set")
            except ValueError:
                log_warning(f"The config parameter 'seeds' must be int or list. (Found: {seeds})")

        # combine sweeps into evaluations
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

        self.num_evaluations = len(self.evaluations) if self.evaluations else 1

    def _start_evaluation(self):
        message = f"Starting Evaluation: {self.current_evaluation + 1} / {self.num_evaluations}"
        warning_message = False
        evaluation_props = {}
        print()

        if self.evaluations:
            # copy params for this evaluation
            self.current_properties = copy.deepcopy(self.properties)
            evaluation_sweep = self.evaluations[self.current_evaluation]
            self.current_sweep = dict(evaluation_sweep)
            evaluation_props["Sweep"] = self.current_sweep

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
        else:
            self.current_properties = self.properties

        # debugging info
        debug_props = ', '.join([k for k in self.properties.keys() if k.startswith("debug_")])
        if self.debugging:
            if len(debug_props) > 0:
                evaluation_props["Debug"] = f"Enabled:  Using options: {debug_props}"
            else:
                evaluation_props["Debug"] = f"Enabled:  ALTHOUGH NO CONFIG OPTIONS DETECTED!"
                warning_message = True

            # catch segfaults
            from glearn.utils.debug import debug_faults
            debug_faults()
        else:
            if len(debug_props) > 0:
                # warn about ignored debug options
                evaluation_props[f"Debug"] = f"Disabled:  IGNORING OPTIONS: {debug_props}"
                warning_message = True

        # log evaluation sweep info
        message = {message: evaluation_props}
        table_color = "yellow" if warning_message else "white"
        print_tabular(message, grouped=True, color=table_color, bold=True, show_type=False)
        print()

        # init session
        self._init_session()

        # prepare random seed
        self.seed = self.get("seed", 1)
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        # config batch size
        self.batch_size = self.get("batch_size", 1)

        # load env or dataset
        self.env = None
        self.dataset = None
        if self.has("env"):
            # make env
            self.env = load_env(self.get("env"))
        elif self.has("dataset"):
            # make dataset
            self.dataset = load_dataset(self)
        if self.env is None and self.dataset is None:
            raise Exception("Failed to find training env or dataset in config")

        # prepare log and save/load paths
        self.log_dir = f"{self.root_log_dir}/{self.current_evaluation + 1}"
        self.summary_path = f"{self.tensorboard_path}/{self.current_evaluation + 1}"
        self.save_path = f"{self.log_dir}/checkpoints/model.ckpt"
        self.load_path = self.save_path

        # create render viewer controller
        self.viewer = load_view_controller(self, render=self.rendering)

        # prepare input/output interfaces, and env
        if self.has_dataset:
            self.input = self.dataset.input
            self.output = self.dataset.output
        elif self.has_env:
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
    def has_env(self):
        return self.env is not None

    @property
    def has_dataset(self):
        return self.dataset is not None

    def get_epoch_size(self, mode="train"):
        if self.has_dataset:
            # dataset epoch size
            return self.dataset.get_epoch_size(mode=mode)
        else:
            # epoch is a single episode (FIXME)
            return 1

    def _init_session(self):
        # set default session
        self.sess = DebuggableSession(self)

        # get global step
        with tf.variable_scope("global_step/"):
            self.global_step = tf.train.get_or_create_global_step()
            self.global_step_update = tf.assign(self.global_step, self.global_step + 1)

    def start_session(self):
        # start checkpoint saving/loading
        if not self._start_checkpoints():
            # initialize all global variables, if not loaded
            self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        if self.sess:
            # stop summary logging and tensorboard
            self._stop_summaries()

            self.sess.close()
            self.sess = None

            # reset default graph
            tf.reset_default_graph()

            if self.env:
                self.env.close()

    def update_global_step(self):
        return self.sess.run(self.global_step_update)

    def _start_summaries(self):
        if self.summary is None:
            if self.summary_path is not None:
                self.summary = SummaryWriter(self)
            else:
                self.summary = NullSummaryWriter()
        self.summary.start()

        # write sweep info summary
        self.write_sweep_summary()

    def write_sweep_summary(self):
        # write any sweep info to tensorboard
        if self.current_sweep is not None:
            with tf.variable_scope("sweep/"):
                tensor = tf.stack([tf.convert_to_tensor([k, str(v)])
                                   for k, v in self.current_sweep.items()])
                self.summary.write_text("hyperparameters", tensor)

    def _stop_summaries(self):
        if self.summary is not None:
            self.summary.stop()

    def _start_checkpoints(self):
        # check if there is a checkpoint
        var_list = None
        load_dir = os.path.dirname(self.load_path)
        load_checkpoint = tf.train.latest_checkpoint(load_dir)

        # check compatibility
        if self.debugging and load_checkpoint:
            model_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
            checkpoint_variables = tf.train.list_variables(load_checkpoint)
            model_var_names = set()
            # var_list = set()
            compatible_var_names = set()
            for model_variable in model_variables:
                model_var_name = model_variable.name.split(":")[0]
                model_var_names.add(model_var_name)
                model_var_shape = model_variable.shape
                for checkpoint_var_name, checkpoint_var_shape in checkpoint_variables:
                    if model_var_name == checkpoint_var_name:
                        if model_var_shape == checkpoint_var_shape:
                            # var_list.add(model_variable)
                            compatible_var_names.add(model_var_name)
                            break
            missing_var_names = model_var_names.difference(compatible_var_names)
            checkpoint_var_names = set([name for name, _ in checkpoint_variables])
            unused_variables = checkpoint_var_names.difference(compatible_var_names)
            if len(missing_var_names) > 0 or len(unused_variables) > 0:
                log_warning(f"\nIncompatible checkpoint file detected: {load_checkpoint}")
                load_checkpoint = None
                if len(missing_var_names) > 0:
                    var_str = "\n * ".join(missing_var_names)
                    log_warning(f"\nMissing model variables from checkpoint:\n * {var_str}")
                if len(unused_variables) > 0:
                    var_str = "\n * ".join(unused_variables)
                    log_warning(f"\nUnused checkpoint variables by model:\n * {var_str}")

        # TODO - Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2, ...)
        self.saver = tf.train.Saver(var_list=var_list)

        # prepare save directory
        if self.save_path is not None:
            save_dir = os.path.dirname(self.save_path)
            os.makedirs(save_dir, exist_ok=True)
            self.dirty_meta_graph = True

        # load any previously saved data for the current version
        return self.load(load_checkpoint)

    def save(self):
        if self.training and self.save_path is not None:
            t0 = time.time()
            save_path = self.saver.save(self.sess, self.save_path, global_step=self.global_step)
            # , write_meta_graph=self.dirty_meta_graph)  # TODO - not sure if I ever need this
            self.dirty_meta_graph = False

            log(f"Saved model: {save_path}  ({time.time() - t0:.2} secs)")
            return True
        return False

    def load(self, load_checkpoint):
        if self.loading and load_checkpoint:
            try:
                log(f"Loading model: {load_checkpoint}")

                self.saver.restore(self.sess, load_checkpoint)
                return True
            except Exception as e:
                log_error(f"Failed to load model: {e}")
        return False


class Configurable(Loggable):
    def __init__(self, config):
        self.config = config

    @property
    def local(self):
        return self.config.local

    @property
    def debugging(self):
        return self.config.debugging

    def is_debugging(self, key):
        return self.config.is_debugging(key)

    @property
    def training(self):
        return self.config.training

    @property
    def dataset(self):
        return self.config.dataset

    @property
    def env(self):
        return self.config.env

    @property
    def has_dataset(self):
        return self.config.has_dataset

    @property
    def has_env(self):
        return self.config.has_env

    @property
    def batch_size(self):
        return self.config.batch_size

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

    @property
    def global_step(self):
        return self.config.global_step

    @property
    def viewer(self):
        return self.config.viewer

    @property
    def rendering(self):
        return self.config.rendering

    def save(self):
        self.config.save()


def load_config(identifier, search_defaults=True, **kwargs):
    # locate the desired config
    config_path = find_config(identifier, search_defaults=search_defaults)
    if config_path is None or not os.path.exists(config_path):
        raise ValueError(f"Failed to locate config using identifier: '{identifier}'")

    return Config(config_path, **kwargs)


def list_configs():
    root = script_relpath("../../configs")
    files = os.listdir(root)
    configs = []
    for f in files:
        if os.path.isfile(os.path.join(root, f)) and f[0] != "_":
            name, ext = os.path.splitext(f)
            if ext in CONFIG_EXTENSIONS:
                configs.append(name)
    configs.sort()
    return configs


def find_config(identifier, search_defaults=True):
    # first try the identifier as if is the full path
    options = [identifier]

    if search_defaults:
        # does it need an extension?
        _, ext = os.path.splitext(identifier)
        if len(ext) == 0:
            options += [f"{identifier}{ext}" for ext in CONFIG_EXTENSIONS]

        # is it relative to the project root?
        root = script_relpath("../../configs")
        options += [os.path.join(root, p) for p in options]

    for path in options:
        if os.path.exists(path):
            return path
    return None
