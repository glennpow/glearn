import os
import sys
import time
from collections import abc
import numpy as np
import tensorflow as tf
import pyglet
import gym
from glearn.policies.interface import Interface
from glearn.datasets import load_dataset
from glearn.datasets.dataset import Transition, transition_batch
from glearn.utils.printing import colorize, print_tabular
from glearn.utils.profile import open_profile
from glearn.utils.reflection import get_class


TEMP_DIR = "/tmp/glearn"


class Policy(object):
    def __init__(self, config, version=None):
        self.config = config

        # get env or dataset
        self.env = None
        self.dataset = None
        if "env" in config:
            # make env
            env_name = config["env"]
            if ":" in env_name:
                # use EntryPoint to get env
                EnvClass = get_class(env_name, config.get("env_args", None))
                self.env = EnvClass()
            elif "-v" in env_name:
                # use gym to get env
                self.env = gym.make(env_name)
                # TODO - pass config["env_args"] to env
            else:
                raise Exception(f"Unrecognizable environment identifier: {env_name}")
            self.project = env_name
        elif "dataset" in config:
            # make dataset
            self.dataset = load_dataset(config)
            self.project = self.dataset.name
        if self.env is None and self.dataset is None:
            raise Exception("Failed to find env or dataset to train with")

        # get basic params
        self.learning_rate = config.get("learning_rate", 1)  # lamdba λ
        self.batch_size = config.get("batch_size", 1)
        self.seed = config.get("seed", 0)
        self.multithreaded = config.get("multithreaded", False)

        # create render viewer
        self.viewer = None
        can_render = sys.stdout.isatty()
        if can_render:
            if "viewer" in config:
                ViewerClass = get_class(config["viewer"])
                self.viewer = ViewerClass()

        # prepare input/output interfaces
        if self.supervised:
            self.input = self.dataset.input
            self.output = self.dataset.output
        elif self.reinforcement:
            self.env.seed(self.seed)

            if self.viewer is not None:
                self.env.unwrapped.viewer = self.viewer

            self.input = Interface(self.env.observation_space)
            self.output = Interface(self.env.action_space)

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

        self.feeds = {}
        self.fetches = {}
        self.layers = {}
        self.results = {}
        self.training = False

        self.observation = None
        self.transitions = []
        self.episode_reward = 0

        if self.viewer is not None:
            self.init_viewer()
        self.init_model()

    def log(self, *args):
        print(*args)

    def error(self, message):
        self.log(colorize(message, "red"))

    def init_viewer(self):
        # register for events from viewer
        self.viewer.window.push_handlers(self)

    def init_model(self):
        # override
        pass

    def start_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.start_persistence()
        self.start_tensorboard()
        self.start_threading()

    def stop_session(self):
        self.stop_threading()

        self.sess.close()

    def start_persistence(self):
        # TODO - only do all this the first time...
        if self.save_path is not None or self.load_path is not None:
            self.saver = tf.train.Saver()

        if self.load_path is not None:
            if os.path.exists(f"{self.load_path}.index"):
                try:
                    self.log(f"Loading model: {self.load_path}")
                    self.saver.restore(self.sess, self.load_path)
                except Exception as e:
                    self.error(str(e))

        if self.save_path is not None:
            self.log(f"Preparing to save model: {self.save_path}")
            save_dir = os.path.dirname(self.save_path)
            os.makedirs(save_dir, exist_ok=True)
            # TODO - clear old dir?

    def start_tensorboard(self):
        if self.tensorboard_path is not None:
            self.log(f"Tensorboard log directory: {self.tensorboard_path}")
            tf.summary.FileWriter(self.tensorboard_path, self.sess.graph)

    def start_threading(self):
        if self.multithreaded:
            # start thread queue
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def stop_threading(self):
        if self.multithreaded:
            # join all threads
            self.coord.request_stop()
            self.coord.join(self.threads)

    @property
    def reinforcement(self):
        return self.env is not None

    @property
    def supervised(self):
        return self.dataset is not None

    def set_feed(self, name, value, graphs=None):
        # set feed node, for graph or global (None)
        if graphs is None:
            # global graph feed
            graphs = ["*"]
        if not isinstance(graphs, list):
            graphs = [graphs]
        # apply to specified graphs
        for graph in graphs:
            if graph in self.feeds:
                graph_feeds = self.feeds[graph]
            else:
                graph_feeds = {}
                self.feeds[graph] = graph_feeds
            graph_feeds[name] = value

    def get_feed(self, name, graph=None):
        # find feed node for graph name
        graph_feeds = self.get_feeds(graph)
        if name in graph_feeds:
            return graph_feeds[name]
        return None

    def get_feeds(self, graph=None):
        # get all global feeds
        feeds = self.feeds.get("*", {})
        if graph is not None:
            # merge with desired graph feeds
            feeds.update(self.feeds.get(graph, {}))
        return feeds

    def build_feed_dict(self, mapping, graph=None):
        feeds = self.get_feeds(graph)
        feed_dict = {}
        for key, value in mapping.items():
            if key in feeds:
                feed = feeds[key]
                feed_dict[feed] = value
            else:
                graph_name = "GLOBAL" if graph is None else graph
                self.error(f"Failed to find feed '{key}' for graph '{graph_name}'")
        return feed_dict

    def set_fetch(self, name, value, graphs=None):
        # set fetch, for graph or global (None)
        if graphs is None:
            # global graph fetch
            graphs = ["*"]
        if not isinstance(graphs, list):
            graphs = [graphs]
        # apply to specified graphs
        for graph in graphs:
            if graph in self.fetches:
                graph_fetches = self.fetches[graph]
            else:
                graph_fetches = {}
                self.fetches[graph] = graph_fetches
            graph_fetches[name] = value

    def get_fetch(self, name, graph=None):
        # find feed node for graph name
        graph_fetches = self.get_fetches(graph)
        if name in graph_fetches:
            return graph_fetches[name]
        return None

    def get_fetches(self, graph):
        # get all global fetches
        fetches = self.fetches.get("*", {})
        if graph != "*":
            # merge with desired graph fetches
            fetches.update(self.fetches.get(graph, {}))
        return fetches

    def add_layer(self, type_name, layer):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
        else:
            type_layers = []
            self.layers[type_name] = type_layers
        type_layers.append(layer)

    def get_layer(self, type_name, index=0):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
            if index < len(type_layers):
                return type_layers[index]
        return None

    def get_layer_count(self, type_name=None):
        if type_name is None:
            return len(self.layers)
        else:
            if type_name in self.layers:
                return len(self.layers[type_name])
        return 0

    def reset(self):
        if self.env is not None:
            self.observation = self.env.reset()
            self.transitions = []
            self.episode_reward = 0

    def render(self, mode="human"):
        if self.env is not None:
            self.env.render(mode=mode)
        if self.viewer is not None:
            self.viewer.render()

    def create_default_feeds(self):
        if self.supervised:
            inputs = self.dataset.get_inputs()
            outputs = self.dataset.get_outputs()
        else:
            inputs = tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")
            outputs = tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")
        self.set_feed("X", inputs)
        self.set_feed("Y", outputs)
        return inputs, outputs

    def prepare_feed_map(self, graph, data, feed_map):
        return feed_map

    def run(self, graph, data, feed_map):
        fetches = self.get_fetches(graph)
        if len(fetches) > 0:
            feed_dict = self.build_feed_dict(feed_map, graph=graph)
            results = self.sess.run(fetches, feed_dict)
            self.results[graph] = results
            return results
        return {}

    def predict(self, data):
        feed_map = self.prepare_feed_map("predict", data, {"X": [data]})
        result = self.run("predict", data, feed_map)["predict"]
        # you only need the first result of batched results
        result = result[0]
        return result

    def rollout(self):
        # get action
        action = self.predict(self.observation)

        # perform action
        new_observation, reward, done, info = self.env.step(self.output.decode(action))

        # record transition
        transition = Transition(self.observation, action, reward, new_observation, done, info)
        self.transitions.append(transition)

        # update stats
        self.observation = new_observation
        self.episode_reward += transition.reward
        return transition

    def get_step_data(self, graph):
        if self.supervised:
            # supervised batch of samples
            return self.dataset.get_step_data()
        else:
            # unsupervised experience replay batch of samples
            batch = transition_batch(self.transitions[:self.batch_size])
            feed_map = {
                "X": batch.inputs,
                "Y": batch.outputs,
            }
            return batch, feed_map

    def optimize(self, step):
        """
        Optimize/evaluate using a supervised or unsupervised batch
        """
        self.step = step
        evaluate_interval = self.config.get("evaluate_interval", 10)
        self.evaluating = step % evaluate_interval == 0

        # log evaluation of current step
        if self.evaluating:
            if self.supervised:
                tab_content = f"  Epoch: {self.epoch}  |  Batch Step: {step}  "
                print(f"\n,{'-' * len(tab_content)},")
                print(f"|{tab_content}|")
            else:
                tab_content = f"  Episode: {self.episode}"
                print(f"\n,{'-' * len(tab_content)},")
                print(f"|{tab_content}|")

            # TODO - print timing info

        # get data and feed and run optimize step pass
        data, feed_map = self.get_step_data("optimize")
        feed_map = self.prepare_feed_map("optimize", data, feed_map)
        optimize_results = self.run("optimize", data, feed_map)

        # evaluate periodically
        if self.evaluating:
            # get feed and run evaluate pass
            feed_map = self.prepare_feed_map("evaluate", data, feed_map)
            evaluate_results = self.run("evaluate", data, feed_map)

            print_tabular(evaluate_results)

            # save model
            if self.save_path is not None:
                save_path = self.saver.save(self.sess, self.save_path)
                self.log(f"Saved model: {save_path}")

        return data, optimize_results

    def train(self, render=False, profile=False):
        episode_rewards = []
        self.training = True
        self.paused = False

        # print training info
        self.print_info()

        def train_yield():
            while True:
                if not self.training:
                    return True

                if render:
                    self.render()

                if self.paused:
                    time.sleep(0)
                else:
                    break
            return False

        def train_loop():
            # start TF session
            self.start_session()

            if self.supervised:
                # supervised learning
                epochs = self.config.get("epochs", 100)

                for epoch in range(epochs):
                    self.dataset.reset()
                    self.epoch = epoch

                    for step in range(self.dataset.epoch_size):
                        self.optimize(step)

                        if train_yield():
                            return
            else:
                # reinforcement learning
                episodes = self.config.get("episodes", 1000)
                max_episode_time = self.config.get("max_episode_time", None)
                min_episode_reward = self.config.get("min_episode_reward", None)

                for episode in range(episodes):
                    self.episode = episode
                    self.reset()
                    tic = time.time()
                    step = 0

                    while True:
                        if train_yield():
                            return

                        # rollout
                        transition = self.rollout()
                        done = transition.done

                        # episode time
                        toc = time.time()
                        elapsed_sec = toc - tic
                        if max_episode_time is not None:
                            # episode timeout
                            if elapsed_sec > max_episode_time:
                                done = True

                        # episode performance
                        episode_reward = self.episode_reward
                        if min_episode_reward is not None:
                            # episode poor performance
                            if episode_reward < min_episode_reward:
                                done = True

                        if done:
                            episode_rewards.append(self.episode_reward)
                            max_reward_so_far = np.amax(episode_rewards)

                            # optimize after episode
                            self.optimize(step)
                            step += 1

                            print_tabular({
                                "episode": episode,
                                "time": elapsed_sec,
                                "reward": episode_reward,
                                "max_reward": max_reward_so_far,
                            })
                            break
            # stop TF session
            self.stop_session()

        if profile:
            profile_path = f"{self.log_dir}/profile"
            with tf.contrib.tfprof.ProfileContext(profile_path) as pctx:  # noqa
                train_loop()
            open_profile(profile_path)
        else:
            train_loop()

    def print_info(self):
        if self.supervised:
            training_info = {
                "Training Method": "Supervised",
                "Dataset": self.dataset,
                "Input": self.dataset.input,
                "Output": self.dataset.output,
                # TODO - get extra subclass stats
            }
        else:
            training_info = {
                "Training Method": "Reinforcement",
                # TODO...
            }
        print()
        print_tabular(training_info, show_type=False)
        print()

    def process_image(self, values, rows=None, cols=None, chans=None):
        # get image dimensions
        values_dims = len(values.shape)
        if values_dims == 1:
            vrows = 1
            vcols = values.shape[0]
            vchans = 1
        elif values_dims == 2:
            vrows, vcols = values.shape
            vchans = 1
        elif values_dims == 3:
            vrows, vcols, vchans = values.shape
        else:
            self.error(f"Too many dimensions ({values_dims} > 3) on passed image data")
            return values

        # get final rows/cols
        if rows is None:
            rows = vrows
        if cols is None:
            cols = vcols

        # init channel mapping
        if isinstance(chans, int):
            chans = range(chans)
        elif isinstance(chans, abc.Iterable):
            pass
        else:
            chans = range(vchans)
        nchans = len(chans)

        # create processed image
        processed = np.zeros((rows, cols, nchans))

        # calculate value ranges, extract channels and normalize
        flat_values = values.ravel()
        size = len(flat_values)
        value_min = min(flat_values)
        value_max = max(flat_values)
        value_range = max([0.1, value_max - value_min])
        flat_values = [int((v - value_min) / value_range * 255) for v in flat_values]
        done = False
        for y in range(rows):
            if done:
                break
            for x in range(cols):
                if done:
                    break
                for c in range(nchans):
                    idx = y * vcols + x + chans[c]
                    if idx >= size:
                        done = True
                        break
                    value = flat_values[idx]
                    processed[y][x][c] = value
        return processed

    def get_viewer_size(self):
        if self.viewer is not None:
            return (int(self.viewer.width), int(self.viewer.height))
        return (0, 0)

    def set_main_image(self, values):
        if self.viewer is not None:
            self.viewer.set_main_image(values)

    def add_image(self, name, values, **kwargs):
        if self.viewer is not None:
            self.viewer.add_image(name, values, **kwargs)

    def remove_image(self, name):
        if self.viewer is not None:
            self.viewer.remove_image(name)

    def add_label(self, name, values, **kwargs):
        if self.viewer is not None:
            self.viewer.add_label(name, values, **kwargs)

    def remove_label(self, name):
        if self.viewer is not None:
            self.viewer.remove_label(name)

    def on_key_press(self, key, modifiers):
        # feature visualization keys
        if key == pyglet.window.key.ESCAPE:
            self.log("Training cancelled by user")
            self.viewer.close()
            self.training = False
        elif key == pyglet.window.key.SPACE:
            self.log(f"Training {'unpaused' if self.paused else 'paused'} by user")
            self.paused = not self.paused
