import os
import time
import atexit
import random
import numpy as np
import tensorflow as tf
import pyglet
from glearn.datasets.dataset import Transition, transition_batch
from glearn.utils.collections import intersects
from glearn.utils.config import Configurable
from glearn.utils.printing import print_update, print_tabular
from glearn.utils.profile import run_profile, open_profile


class Trainer(Configurable):
    def __init__(self, config, policy, epochs=100, episodes=1000, max_episode_time=None,
                 min_episode_reward=None, evaluate_interval=10, epsilon=0, keep_prob=1, **kwargs):
        super().__init__(config)

        self.policy = policy
        self.kwargs = kwargs

        self.batch_size = config.get("batch_size", 1)
        self.debug_gradients = self.config.get("debug_gradients", False)

        self.epochs = epochs
        self.episodes = episodes
        self.max_episode_time = max_episode_time
        self.min_episode_reward = min_episode_reward
        self.evaluate_interval = evaluate_interval
        self.epsilon = epsilon
        self.keep_prob = keep_prob

        self.epoch_step = 0
        self.episode_step = 0
        self.observation = None
        self.transitions = []
        self.episode_reward = 0
        self.batch = None

        # get global step
        self.global_step = tf.train.get_or_create_global_step()

        if self.rendering:
            self.init_viewer()
        self.init_optimizer()

    def __str__(self):
        properties = [
            "supervised" if self.supervised else "reinforcement",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    @property
    def save_path(self):
        return self.config.save_path

    @property
    def load_path(self):
        return self.config.load_path

    def start_session(self):
        self.config.start_session()
        self.policy.start_session()

        self.start_persistence()

    def stop_session(self):
        self.policy.stop_session()
        self.config.stop_session()

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

    def reset(self):
        # reset env and episode
        if self.env is not None:
            self.observation = self.env.reset()
            self.transitions = []
            self.episode_reward = 0

    def optimize_loss(self, loss, graph="optimize", definition=None):
        # default definition
        if definition is None:
            definition = self.kwargs

        global_step = self.global_step

        with tf.name_scope(graph):
            learning_rate = definition.get("learning_rate", 1e-2)

            # learning rate decay
            lr_decay = definition.get("lr_decay", None)
            if lr_decay is not None:
                lr_decay_epochs = definition.get("lr_decay_epochs", 1)
                epoch_size = self.dataset.get_epoch_size(partition="train")
                decay_steps = int(lr_decay_epochs * epoch_size)
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,
                                                           lr_decay, staircase=True)

            # create optimizer
            optimizer_name = definition.get("optimizer", "sgd")
            if optimizer_name == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif optimizer_name == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            else:
                raise Exception(f"Unknown optimizer type specified in config: {optimizer_name}")

            # get gradients and trainable variables
            grads_tvars = optimizer.compute_gradients(loss)

            # check if we require unzipping grad/vars
            max_grad_norm = definition.get("max_grad_norm", None)
            require_unzip = self.debug_gradients or max_grad_norm is not None
            if require_unzip:
                grads, tvars = zip(*grads_tvars)

            # apply gradient clipping
            if max_grad_norm is not None:
                clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

                # metric to observe clipped gradient ratio
                unequal = [tf.reduce_mean(tf.cast(tf.not_equal(grad, clipped_grad), tf.float32))
                           for grad, clipped_grad in list(zip(grads, clipped_grads))]
                clipped_ratio = tf.reduce_mean(unequal)

                grads = clipped_grads

            if require_unzip:
                grads_tvars = zip(grads, tvars)

            # apply gradients
            optimize = optimizer.apply_gradients(grads_tvars, global_step=global_step)

            self.policy.set_fetch(graph, optimize)  # HACK - does this work for actor-critic?

        # add learning rate and gradient summaries
        self.summary.add_scalar("learning_rate", learning_rate, graph)
        if self.debug_gradients:
            self.summary.add_gradients(zip(grads, tvars), "debug")
        if max_grad_norm is not None:
            self.summary.add_scalar("clipped_ratio", clipped_ratio, graph)

        return optimize

    def init_optimizer(self):
        # get accuracy summary from policy
        accuracy = self.policy.get_fetch("accuracy", "evaluate")
        if accuracy is not None:
            self.summary.add_scalar("accuracy", accuracy, "evaluate")

    def prepare_feeds(self, graphs, feed_map):
        self.policy.prepare_default_feeds(graphs, feed_map)

        # dropout
        if intersects(["policy_optimize", "value_optimize"], graphs):
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1

        return feed_map

    def run(self, graphs, feed_map={}, render=True):
        if not isinstance(graphs, list):
            graphs = [graphs]

        # run policy for graph with feeds
        feed_map = self.prepare_feeds(graphs, feed_map)
        results = self.policy.run(graphs, feed_map)

        # view results
        if render:
            self.viewer.view_results(graphs, feed_map, results)

        return results

    def fetch(self, name, feed_map={}, squeeze=False):
        # shortcut to fetch a single graph/value
        results = self.run(name, feed_map)
        fetch_result = results[name]

        if squeeze:
            fetch_result = np.squeeze(fetch_result)
        return fetch_result

    def predict(self, inputs):
        # input as feed map
        feed_map = {"X": [inputs]}

        # get desired graphs
        graphs = ["predict"]
        if self.debugging:
            graphs.append("debug")

        # evaluate graphs and extract single prediction
        results = self.run(graphs, feed_map)
        return results["predict"][0]

    def action(self):
        # decaying epsilon-greedy
        epsilon = self.epsilon
        if isinstance(epsilon, list):
            # FIXME - should this be per current_global_iteration instead of current_global_step?
            t = min(1, self.current_global_step / epsilon[2])
            epsilon = t * (epsilon[1] - epsilon[0]) + epsilon[0]

        # get action
        if np.random.random() < epsilon:
            # choose epsilon-greedy random action  (TODO - could implement this in tf)
            return self.output.sample()
        else:
            # choose optimal policy action
            return self.predict(self.observation)

    def rollout(self):
        # get action
        action = self.action()

        # perform action
        new_observation, reward, done, info = self.env.step(self.output.decode(action))

        # build and process transition
        transition = Transition(self.observation, action, reward, new_observation, done, info)
        self.process_transition(transition)

        # record transition
        self.transitions.append(transition)

        # update stats
        self.observation = new_observation
        self.episode_reward += transition.reward
        return transition

    def process_transition(self, transition):
        pass

    def get_batch(self):
        if self.supervised:
            # prepare supervised batch
            return self.dataset.get_batch()
        else:
            # unsupervised experience replay batch of samples
            batch = transition_batch(self.transitions[:self.batch_size])
            feed_map = {
                "X": batch.inputs,
                "Y": batch.outputs,
            }
            return batch, feed_map

    def pre_optimize(self, feed_map):
        pass

    def post_optimize(self, feed_map):
        pass

    def optimize(self):
        # get either supervised or unsupervised batch data and feeds
        self.batch, feed_map = self.get_batch()

        # run all desired graphs
        results = self.run(["policy_optimize"], feed_map)

        # get current global step (TODO: add global_step fetch into above run)
        global_step = tf.train.global_step(self.sess, self.global_step)
        global_step += 1  # HACK: +1, this doesn't seem to be updated yet
        self.current_global_step = global_step

        # print log
        iteration_name = "Epoch" if self.supervised else "Episode"
        iteration = self.epoch if self.supervised else self.episode
        step = self.epoch_step if self.supervised else self.episode_step
        print_update(f"Optimizing | {iteration_name}: {iteration} | Step: {step} | "
                     f"Global Step: {global_step} | Eval. Steps: {self.evaluate_interval}")
        return results

    @property
    def evaluating(self):
        return self.current_global_step % self.evaluate_interval == 0

    def evaluate(self, train_yield):
        # Evaluate using the test dataset
        graphs = ["evaluate"]
        if self.debugging:
            graphs.append("debug")

        # prepare dataset partition
        if self.supervised:
            epoch_size = self.dataset.initialize(partition="test")
        else:
            epoch_size = 1

        # get batch data and desired graphs
        eval_start_time = time.time()
        averaged_results = {}
        report_step = random.randrange(epoch_size)
        report_results = None
        report_feed_map = None
        for step in range(epoch_size):
            print_update(f"Evaluating | Progress: {step}/{epoch_size}")

            reporting = step == report_step
            self.batch, feed_map = self.get_batch()

            # run evaluate graphs
            results = self.run(graphs, feed_map, render=reporting)

            # gather reporting results
            if reporting:
                report_results = results
                report_feed_map = feed_map
            for k, v in results.items():
                if self.policy.is_fetch(k, "evaluate") and \
                   (isinstance(v, float) or isinstance(v, int)):
                    if k in averaged_results:
                        averaged_results[k] += v
                    else:
                        averaged_results[k] = v

            if train_yield():
                return

        if report_results is not None:
            # log stats for current evaluation
            current_time = time.time()
            train_elapsed_time = current_time - self.train_start_time
            iteration_elapsed_time = current_time - self.iteration_start_time
            eval_elapsed_time = eval_start_time - (self.last_eval_time or self.train_start_time)
            eval_steps = self.current_global_step - (self.last_eval_step or 0)
            steps_per_second = eval_steps / eval_elapsed_time
            self.last_eval_time = current_time
            self.last_eval_step = self.current_global_step
            stats = {
                "global step": self.current_global_step,
                "training time": train_elapsed_time,
                "steps/second": steps_per_second,
            }
            if self.supervised:
                stats.update({
                    "epoch step": self.epoch_step,
                    "epoch time": iteration_elapsed_time,
                })
                table = {f"Epoch: {self.epoch}": stats}
            else:
                stats.update({
                    "episode steps": self.episode_step,
                    "episode time": iteration_elapsed_time,
                    "reward": self.episode_reward,
                    "max reward": self.max_episode_reward,
                })
                table = {f"Episode: {self.episode}": stats}

            # average evaluate results
            averaged_results = {k: v / epoch_size for k, v in averaged_results.items()}
            report_results.update(averaged_results)

            # remove None values
            report_results = {k: v for k, v in report_results.items() if v is not None}

            # print inputs and results
            table["Inputs"] = report_feed_map
            table["Evaluation"] = report_results

            # print tabular results
            print_tabular(table, grouped=True)

            # summaries
            self.summary.add_simple_value("steps_per_second", steps_per_second, "evaluate")

        # save model
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            self.log(f"Saved model: {save_path}")

        print()

    def train(self, render=False, profile=False):
        self.max_episode_reward = None
        self.training = True
        self.paused = False
        self.train_start_time = time.time()

        # check for invalid values in the current graph
        if self.config.get("debug_numerics", False):
            self.policy.set_fetch("check", tf.add_check_numerics_ops(), "debug")

        # prepare viewer
        self.viewer.prepare(self)

        # print training info
        self.print_info()

        # yield after each training iteration
        def train_yield(flush_summary=False):
            # write summary results
            if flush_summary:
                self.policy.summary.flush(global_step=self.current_global_step)

            while True:
                # check if training stopped
                if not self.training:
                    return True

                # render frame
                if render:
                    self.render()

                # loop while paused
                if self.paused:
                    time.sleep(0)
                else:
                    break
            return False

        # main training loop
        def train_loop():
            # start TF session
            self.start_session()

            # cleanup TF session
            def cleanup():
                self.stop_session()
            atexit.register(cleanup)

            # get current global step, and prepare evaluation counters
            global_step = tf.train.global_step(self.sess, self.global_step)
            self.current_global_step = global_step
            self.last_eval_time = None
            self.last_eval_step = self.current_global_step

            # do supervised or reinforcement loop
            if self.supervised:
                self.train_supervised_loop(train_yield)
            else:
                self.train_reinforcement_loop(train_yield)

        if profile:
            # profile training loop
            profile_path = run_profile(train_loop, self.config)

            # show profiling results
            open_profile(profile_path)
        else:
            # run training loop without profiling
            train_loop()

    def train_supervised_loop(self, train_yield):
        # supervised learning
        for epoch in range(self.epochs):
            # start current epoch
            epoch_size = self.dataset.initialize(partition="train")
            self.epoch = epoch
            self.iteration_start_time = time.time()

            # epoch summary (TODO - store this in variable)
            self.summary.add_scalar("epoch", self.global_step / epoch_size, "evaluate")

            for step in range(epoch_size):
                # epoch time
                self.epoch_step = step + 1

                # optimize batch
                self.optimize()

                # evaluate if time to do so
                if self.evaluating:
                    self.evaluate(train_yield)

                if train_yield(True):
                    return

    def train_reinforcement_loop(self, train_yield):
        # reinforcement learning
        for episode in range(self.episodes):
            # start current episode
            self.iteration_start_time = time.time()
            self.episode = episode
            self.reset()
            self.episode_step = 0

            # episode summary (TODO - store this in variable)
            self.summary.add_simple_value("episode", episode, "evaluate")

            while self.training:
                if train_yield(True):
                    return

                # rollout
                transition = self.rollout()
                done = transition.done
                self.episode_step += 1

                # episode time
                current_time = time.time()
                episode_time = current_time - self.iteration_start_time
                if self.max_episode_time is not None:
                    # episode timeout
                    if episode_time > self.max_episode_time:
                        done = True

                # episode performance
                episode_reward = self.episode_reward
                if self.min_episode_reward is not None:
                    # episode poor performance
                    if episode_reward < self.min_episode_reward:
                        done = True

                if done:
                    if self.max_episode_reward is None \
                       or self.episode_reward > self.max_episode_reward:
                        self.max_episode_reward = self.episode_reward

                    # summary values
                    self.summary.add_simple_value("max_episode_reward", self.max_episode_reward,
                                                  "evaluate")

                    # optimize after episode
                    self.optimize()

                    # evaluate if time to do so
                    if self.evaluating:
                        self.evaluate(train_yield)
                    break
            if not self.training:
                return

    def print_info(self):
        if self.supervised:
            training_info = {
                "Dataset": self.dataset,
                "Trainer": self,
                "Policy": self.policy,
                "Input": self.dataset.input,
                "Output": self.dataset.output,
            }
        else:
            training_info = {
                "Environment": self.project,
                "Trainer": self,
                "Policy": self.policy,
                "Input": self.input,
                "Output": self.output,
            }
        print()
        print_tabular(training_info, show_type=False, color="white", bold=True)
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
            self.warning("Training cancelled by user")
            self.viewer.close()
            self.training = False
        elif key == pyglet.window.key.SPACE:
            self.warning(f"Training {'unpaused' if self.paused else 'paused'} by user")
            self.paused = not self.paused
