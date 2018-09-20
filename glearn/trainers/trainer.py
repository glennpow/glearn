import os
import time
import numpy as np
import tensorflow as tf
import pyglet
from glearn.datasets.dataset import Transition, transition_batch
from glearn.utils.printing import colorize, print_tabular
from glearn.utils.profile import open_profile


class Trainer(object):
    def __init__(self, config, policy):
        self.config = config
        self.policy = policy

        self.batch_size = config.get("batch_size", 1)
        self.seed = config.get("seed", 0)
        self.epochs = self.config.get("epochs", 100)
        self.episodes = self.config.get("episodes", 1000)
        self.max_episode_time = self.config.get("max_episode_time", None)
        self.min_episode_reward = self.config.get("min_episode_reward", None)
        self.evaluate_interval = self.config.get("evaluate_interval", 10)

        self.iteration = 0
        self.epoch_step = 0
        self.episode_step = 0
        self.observation = None
        self.transitions = []
        self.episode_reward = 0

        if self.rendering:
            self.init_viewer()
        self.init_optimizer()

    def log(self, *args):
        print(*args)

    def error(self, message):
        self.log(colorize(message, "red"))

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
    def save_path(self):
        return self.config.save_path

    @property
    def load_path(self):
        return self.config.load_path

    @property
    def tensorboard_path(self):
        return self.config.tensorboard_path

    def start_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.policy.start_session(self.sess)

        self.start_persistence()
        self.start_tensorboard()

    def stop_session(self):
        self.policy.stop_session(self.sess)

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

    def reset(self):
        # reset env and episode
        if self.env is not None:
            self.observation = self.env.reset()
            self.transitions = []
            self.episode_reward = 0

    def init_optimizer(self):
        pass

    def prepare_feeds(self, graph, data, feed_map):
        return self.policy.prepare_default_feeds(graph, feed_map)

    def predict(self, data):
        feed_map = self.prepare_feeds("predict", data, {"X": [data]})
        result = self.policy.predict(self.sess, feed_map)["predict"][0]

        # debugging
        if self.debugging:
            feed_map = self.prepare_feeds("debug", data, feed_map)
            self.policy.debug(self.sess, feed_map)

        return result

    def action(self):
        return self.predict(self.observation)

    def rollout(self):
        # get action
        action = self.action()

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

    def optimize(self):
        """
        Optimize/evaluate using a supervised or unsupervised batch
        """
        self.iteration += 1
        self.evaluating = self.iteration % self.evaluate_interval == 0

        # log info for current iteration
        if self.supervised:
            table = {
                f"Epoch: {self.epoch}": {
                    "epoch iteration": self.iteration,
                    "epoch time": self.epoch_time,
                }
            }
        else:
            table = {
                f"Episode: {self.episode}": {
                    "episode steps": self.episode_step,
                    "episode time": self.episode_time,
                    "reward": self.episode_reward,
                    "max reward": np.amax(self.episode_rewards),
                }
            }

        # get data and feed and run optimize step pass
        data, feed_map = self.get_step_data("optimize")
        feed_map = self.prepare_feeds("optimize", data, feed_map)
        optimize_results = self.policy.optimize(self.sess, feed_map)

        # evaluate if time to do so
        if self.evaluating:
            # get feed and run evaluate pass
            feed_map = self.prepare_feeds("evaluate", data, feed_map)
            evaluate_results = self.policy.evaluate(self.sess, feed_map)

            # # debugging
            debug_results = None
            if self.debugging:
                feed_map = self.prepare_feeds("debug", data, feed_map)
                debug_results = self.policy.debug(self.sess, feed_map)

            # print inputs and results
            table["Inputs"] = feed_map
            table["Evaluation"] = evaluate_results
            if debug_results is not None and len(debug_results) > 0:
                table["Debug"] = debug_results

            # print tabular results
            print_tabular(table, grouped=True)
            print()

            # save model
            if self.save_path is not None:
                save_path = self.saver.save(self.sess, self.save_path)
                self.log(f"Saved model: {save_path}")

        return data, optimize_results

    def train(self, render=False, profile=False):
        self.episode_rewards = []
        self.training = True
        self.paused = False

        # print training info
        self.print_info()

        # yield after each training iteration
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

        # main training loop
        def train_loop():
            # start TF session
            self.start_session()

            # do supervised or reinforcement loop
            if self.supervised:
                self.train_supervised_loop(train_yield)
            else:
                self.train_reinforcement_loop(train_yield)

            # stop TF session
            self.stop_session()

        if profile:
            # profile training loop
            profile_path = f"{self.log_dir}/profile"
            with tf.contrib.tfprof.ProfileContext(profile_path) as pctx:  # noqa
                train_loop()
            # show profiling results
            open_profile(profile_path)
        else:
            # run training loop without profiling
            train_loop()

    def train_supervised_loop(self, train_yield):
        # supervised learning
        for epoch in range(self.epochs):
            self.dataset.reset()
            self.epoch = epoch
            tic = time.time()

            for step in range(self.dataset.epoch_size):
                # epoch time
                toc = time.time()
                self.epoch_time = toc - tic
                self.epoch_step = step

                self.optimize()

                if train_yield():
                    return

    def train_reinforcement_loop(self, train_yield):
        # reinforcement learning
        for episode in range(self.episodes):
            # start current episode
            self.episode = episode
            self.reset()
            tic = time.time()
            self.episode_step = 0

            while True:
                if train_yield():
                    return

                # rollout
                transition = self.rollout()
                done = transition.done
                self.episode_step += 1

                # episode time
                toc = time.time()
                self.episode_time = toc - tic
                if self.max_episode_time is not None:
                    # episode timeout
                    if self.episode_time > self.max_episode_time:
                        done = True

                # episode performance
                episode_reward = self.episode_reward
                if self.min_episode_reward is not None:
                    # episode poor performance
                    if episode_reward < self.min_episode_reward:
                        done = True

                if done:
                    self.episode_rewards.append(self.episode_reward)

                    # optimize after episode
                    self.optimize()
                    break

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
                "Environment": self.project,
                "Input": self.input,
                "Output": self.output,
                # TODO...
            }
        print()
        print_tabular(training_info, show_type=False)
        print()

    @property
    def viewer(self):
        return self.config.viewer

    @property
    def rendering(self):
        return self.viewer.rendering

    def init_viewer(self):
        # register for events from viewer
        self.viewer.add_listener(self)

    def render(self, mode="human"):
        if self.env is not None:
            self.env.render(mode=mode)
        self.viewer.render()

    def on_key_press(self, key, modifiers):
        # feature visualization keys
        if key == pyglet.window.key.ESCAPE:
            self.log("Training cancelled by user")
            self.viewer.close()
            self.training = False
        elif key == pyglet.window.key.SPACE:
            self.log(f"Training {'unpaused' if self.paused else 'paused'} by user")
            self.paused = not self.paused
