import os
import time
from collections import abc
import numpy as np
import tensorflow as tf
import pyglet
from policies.interface import Interface
from datasets.dataset import Transition, transition_batch
from utils.viewer import AdvancedViewer
from utils.printing import colorize, print_tabular


class Policy(object):
    def __init__(self, env=None, dataset=None, batch_size=None, seed=0,
                 load_path=None, save_path=None, tensorboard_path=None,
                 multithreaded=False):
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        if self.supervised:
            self.input = dataset.input
            self.output = dataset.output

            self.viewer = AdvancedViewer()
        elif self.reinforcement:
            self.env.seed(self.seed)

            self.viewer = AdvancedViewer()
            self.env.unwrapped.viewer = self.viewer

            self.input = Interface(env.observation_space)
            self.output = Interface(env.action_space)

        self.load_path = load_path
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path
        self.multithreaded = multithreaded

        self.act_graph = {}
        self.optimize_graph = {}
        self.evaluate_graph = {}

        self.act_result = {}
        self.optimize_result = {}
        self.evaluate_result = {}

        self.observation = None
        self.transitions = []
        self.episode_reward = 0

        self.layers = {}
        self.training = False

        self.init_viewer()
        self.init_model()
        self.init_session()
        self.init_persistence()
        self.init_tensorboard()

    def log(self, *args):
        print(*args)

    def error(self, message):
        self.log(colorize(message, "red"))

    def init_model(self):
        pass

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.init_threading()

    def init_persistence(self):
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

    def init_tensorboard(self):
        if self.tensorboard_path is not None:
            self.log(f"Tensorboard log directory: {self.tensorboard_path}")
            tf.summary.FileWriter(self.tensorboard_path, self.sess.graph)

    def init_viewer(self):
        # register for events from viewer
        if self.viewer is not None:
            self.viewer.window.push_handlers(self)

    def init_threading(self):
        if self.multithreaded:
            # start thread queue
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def update_threading(self):
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

    def get_layer_count(self, type_name=None):
        if type_name is None:
            return len(self.layers)
        else:
            if type_name in self.layers:
                return len(self.layers[type_name])
        return 0

    def get_layer(self, type_name, index=0):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
            if index < len(type_layers):
                return type_layers[index]
        return None

    def add_layer(self, type_name, layer):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
        else:
            type_layers = []
            self.layers[type_name] = type_layers
        type_layers.append(layer)

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

    def create_inputs(self):
        if self.supervised:
            self.inputs = self.dataset.get_inputs()
            self.outputs = self.dataset.get_outputs()
        else:
            self.inputs = tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")
            self.outputs = tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")
        return self.inputs, self.outputs

    def get_epoch(self):
        if self.supervised:
            # supervised epoch of samples
            return self.dataset.get_epoch()
        else:
            # unsupervised replay batch samples
            batch = transition_batch(self.transitions[:self.batch_size])
            feed_dict = {
                self.inputs: batch.inputs,
                self.outputs: batch.outputs,
            }
            return batch, feed_dict

    def rollout(self):
        # get action
        action = self.act(self.observation)

        # perform action
        new_observation, reward, done, info = self.env.step(self.output.decode(action))

        # record transition
        transition = Transition(self.observation, action, reward, new_observation, done, info)
        self.transitions.append(transition)

        # update stats
        self.observation = new_observation
        self.episode_reward += transition.reward
        return transition

    def act_feed(self, observation, feed_dict):
        return feed_dict

    def act(self, observation):
        if "act" in self.act_graph:
            # prepare parameters
            feed_dict = self.act_feed(observation, {self.inputs: [observation]})

            # evaluate act graph
            self.act_result = self.sess.run(self.act_graph, feed_dict=feed_dict)
            action = self.act_result["act"]

            # join threads
            self.update_threading()

            return action.ravel()
        return np.zeros(self.output.shape)

    def optimize_feed(self, data, feed_dict):
        return feed_dict

    def optimize(self, evaluating=False, saving=True):
        """
        Run an entire supervised epoch, or batch of unsupervised episodes.
        """
        if "optimize" in self.optimize_graph:
            # get data for epoch
            data, feed_dict = self.get_epoch()
            feed_dict = self.optimize_feed(data, feed_dict)

            # evaluate optimize graph
            self.optimize_result = self.sess.run(self.optimize_graph, feed_dict=feed_dict)

            # save model
            if saving and self.save_path is not None:
                save_path = self.saver.save(self.sess, self.save_path)
                self.log(f"Saved model: {save_path}")

            # join threads
            self.update_threading()

            # evaluate periodically
            if evaluating and len(self.evaluate_graph) > 0:
                # prepare evaluation parameters
                feed_dict = self.act_feed(data, feed_dict)

                # run evaluate graph
                self.evaluate_result = self.sess.run(self.evaluate_graph, feed_dict=feed_dict)

                print_tabular(self.evaluate_result)

                # join threads
                self.update_threading()
            return data
        return None

    def train(self, episodes, epochs=1, max_episode_time=None, min_episode_reward=None,
              render=False, evaluate_interval=20, profile_path=None):
        episode_rewards = []
        self.training = True

        # print training info
        self.print_info()

        def train_loop():
            if self.supervised:
                # supervised learning
                for epoch in range(epochs):
                    if not self.training:
                        return

                    self.dataset.reset()

                    evaluating = epoch % evaluate_interval == 0
                    saving = evaluating
                    self.optimize(evaluating=evaluating, saving=saving)

                    if render:
                        self.render()
            else:
                # reinforcement learning
                for episode in range(episodes):
                    self.reset()
                    tic = time.time()

                    while True:
                        if not self.training:
                            return

                        if render:
                            self.render()

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
                            self.optimize()

                            print_tabular({
                                "episode": episode,
                                "time": elapsed_sec,
                                "reward": episode_reward,
                                "max_reward": max_reward_so_far,
                            })
                            break

        if profile_path is not None:
            with tf.contrib.tfprof.ProfileContext(profile_path) as pctx:  # noqa
                train_loop()
        else:
            train_loop()

    def print_info(self):
        if self.supervised:
            training_info = {
                "Training Method": "Supervised",
                "Dataset": self.dataset.name,
                "Input": self.dataset.input,
                "Output": self.dataset.output,
                "Batch Size": self.dataset.batch_size,
            }
        else:
            training_info = {
                "Training Method": "Reinforcement",
                # TODO...
            }
        print_tabular(training_info)

    def get_viewer_size(self):
        if self.viewer is not None:
            return (self.viewer.width, self.viewer.height)
        return (0, 0)

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
        mrows = min(rows, vrows)
        mcols = min(cols, vcols)

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
        value_min = min(flat_values)
        value_max = max(flat_values)
        value_range = max([0.1, value_max - value_min])
        for y in range(mrows):
            for x in range(mcols):
                for c in range(nchans):
                    idx = y * vcols + x + chans[c]
                    value = flat_values[idx]
                    norm = int((value - value_min) / value_range * 255)
                    processed[y][x][c] = norm
        return processed

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
            self.viewer.close()
            self.training = False
